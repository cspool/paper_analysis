import os
import sys
import re
import shutil
import argparse
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

OCR_BLOCK_BEFORE_IMAGE_RE = re.compile(
    r'(?:^|\n)'
    r'> \*\*\[图片提取文字 [^\]\n]*\](?::\*\*|\*\*:)\n'
    r'(?:> ?[^\n]*(?:\n|$))+'
    r'\s*$'
)


@dataclass
class ImageMatch:
    start: int
    end: int
    alt: str
    src: str
    img_tag: str


def is_escaped(text, idx):
    """判断 text[idx] 是否被反斜杠转义。"""
    backslash_count = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslash_count += 1
        cursor -= 1
    return backslash_count % 2 == 1


def find_unescaped(text, target, start):
    cursor = start
    while cursor < len(text):
        if text[cursor] == target and not is_escaped(text, cursor):
            return cursor
        cursor += 1
    return -1


def find_closing_paren(text, start):
    depth = 0
    cursor = start
    while cursor < len(text):
        char = text[cursor]
        if is_escaped(text, cursor):
            cursor += 1
        elif char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return cursor
            depth -= 1
        cursor += 1
    return -1


def iter_markdown_images(content):
    """遍历 Markdown 图片语法，支持图片路径中出现未转义括号。"""
    search_start = 0
    while True:
        start = content.find("![", search_start)
        if start == -1:
            break

        alt_start = start + 2
        alt_end = find_unescaped(content, "]", alt_start)
        if alt_end == -1:
            break

        if alt_end + 1 >= len(content) or content[alt_end + 1] != "(":
            search_start = alt_end + 1
            continue

        src_start = alt_end + 2
        src_end = find_closing_paren(content, src_start)
        if src_end == -1:
            break

        yield ImageMatch(
            start=start,
            end=src_end + 1,
            alt=content[alt_start:alt_end],
            src=content[src_start:src_end].strip(),
            img_tag=content[start:src_end + 1],
        )
        search_start = src_end + 1


def normalize_markdown_src(src):
    src = src.strip()
    if src.startswith("<") and src.endswith(">"):
        return src[1:-1]
    return src


def has_existing_ocr_block(content, img_start):
    return OCR_BLOCK_BEFORE_IMAGE_RE.search(content[:img_start]) is not None


def shell_single_quote(value):
    return "'" + value.replace("'", "'\\''") + "'"


def run_marker_ocr(img_path_str, md_file_dir, torch_device="cuda"):
    """
    通过 subprocess 的 bash 环境调用 marker_single 执行 OCR，并提取生成的 markdown 文本
    """
    # 1. 解析图片的绝对路径（处理相对路径问题），URL解码处理Notion导出的编码路径
    img_path = Path(unquote(normalize_markdown_src(img_path_str)))
    if not img_path.is_absolute():
        abs_img_path = (Path(md_file_dir) / img_path).resolve()
    else:
        abs_img_path = img_path.resolve()

    if not abs_img_path.exists():
        print(f"    [跳过] 图片文件不存在: {abs_img_path}")
        return None

    # 2. 创建临时目录接收 marker 输出，避免污染工作区
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 对路径进行安全的单引号转义，防止 Bash 注入或空格截断
        safe_img_path = shell_single_quote(str(abs_img_path))
        safe_tmp_dir = shell_single_quote(str(tmp_dir))
        
        # 组装符合 marker 规范的 bash 命令
        bash_command = (
            f"marker_single {safe_img_path} "
            f"--output_dir {safe_tmp_dir} "
            f"--disable_multiprocessing"
        )
        marker_env = os.environ.copy()
        marker_env["TORCH_DEVICE"] = torch_device
        
        try:
            # 显式调用 /bin/bash 执行命令，设置 120 秒超时防卡死
            result = subprocess.run(
                bash_command,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=120,
                env=marker_env,
            )
            
            if result.returncode != 0:
                print(f"    [错误] Marker 运行失败 (Exit Code {result.returncode})")
                print(f"    [错误日志] {result.stderr.strip()}")
                return None
            
            # 3. 扫描临时目录，获取 marker 生成的 .md 结果文件
            generated_mds = list(Path(tmp_dir).glob("**/*.md"))
            if not generated_mds:
                print(f"    [警告] Marker 正常结束，但未检测到生成的 Markdown 结果")
                return None
            
            # 读取 OCR 提取出来的文本内容
            ocr_text = generated_mds[0].read_text(encoding='utf-8').strip()
            return ocr_text

        except subprocess.TimeoutExpired:
            print(f"    [超时] 该图片 OCR 处理超时（超过 120 秒），已自动跳过")
            return None
        except Exception as e:
            print(f"    [异常] 执行过程中出现系统错误: {e}")
            return None

def process_md_file(md_path, skip_backup=False, torch_device="cuda", remove_images=False):
    """
    处理单个 Markdown 文件，解析图片、调用 OCR 并安全回写
    """
    try:
        content = md_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"[文件读取失败] {md_path}: {e}")
        return

    md_dir = md_path.parent
    new_content = ""
    last_idx = 0
    updated_count = 0
    skipped_count = 0
    images_to_remove = []  # 记录成功 OCR 的图片绝对路径，用于后续可选删除
    
    # 查找文章中所有的图片引用
    matches = list(iter_markdown_images(content))
    if not matches:
        return

    print(f"\n[正在处理] {md_path} (共发现 {len(matches)} 处图片引用)")

    for match in matches:
        start, end = match.start, match.end
        # 拼接上一个匹配项到当前匹配项之间的原生文本
        new_content += content[last_idx:start]
        
        img_tag = match.img_tag
        alt = match.alt
        src = normalize_markdown_src(match.src)
        
        # 核心恢复机制：如果图片前面已经包含 OCR 标记块，说明是断点恢复，直接保留并跳过
        if has_existing_ocr_block(content, start):
            new_content += img_tag
            skipped_count += 1
            last_idx = end
            continue
            
        # 跳过网络图片链接，只处理本地文件
        if src.startswith(('http://', 'https://')):
            new_content += img_tag
            last_idx = end
            continue

        print(f"  --> 正在 OCR 图片: {src}")
        ocr_result = run_marker_ocr(src, md_dir, torch_device=torch_device)
        
        if ocr_result:
            # 格式化文本，使其在 Markdown 中表现为优美的引用块 (Blockquote)
            indented_ocr = "\n> ".join(ocr_result.split("\n"))
            formatted_block = (
                f"> **[图片提取文字 ({alt or '无描述'})]:**\n"
                f"> {indented_ocr}\n"
            )
            # 将 OCR 结果拼在图片原本标签的前面
            new_content += formatted_block + img_tag
            updated_count += 1
            # 记录图片绝对路径（用于可选删除）
            img_path = Path(unquote(normalize_markdown_src(src)))
            if not img_path.is_absolute():
                abs_img_path = (Path(md_dir) / img_path).resolve()
            else:
                abs_img_path = img_path.resolve()
            images_to_remove.append(abs_img_path)
        else:
            # OCR 失败则保持原样
            new_content += img_tag
            
        last_idx = end
        
    new_content += content[last_idx:]

    # 如果有新图片被处理，执行安全回写
    if updated_count > 0:
        # 创建备份文件 (.bak) 防止写入中断导致原文档损坏
        if not skip_backup:
            bak_path = md_path.with_suffix(md_path.suffix + '.bak')
            shutil.copy2(md_path, bak_path)
            
        # 写入新内容
        md_path.write_text(new_content, encoding='utf-8')
        print(f"[完成] 成功为 {updated_count} 张图片注入描述 (跳过已处理/历史图片: {skipped_count} 张)")

        # 如果指定了 --remove-images，删除已成功 OCR 的原始图片
        if remove_images and images_to_remove:
            removed_count = 0
            for img_path in images_to_remove:
                try:
                    if img_path.exists():
                        img_path.unlink()
                        print(f"  [已删除] 原始图片: {img_path}")
                        removed_count += 1
                    else:
                        print(f"  [跳过删除] 图片不存在: {img_path}")
                except Exception as e:
                    print(f"  [删除失败] {img_path}: {e}")
            if removed_count > 0:
                print(f"[清理] 已删除 {removed_count} 张原始图片")
    else:
        print(f"[未变更] 无需更新任何新图片 (已处理/历史图片: {skipped_count} 张)")

def main():
    parser = argparse.ArgumentParser(description="批量读取Markdown文件中的图片并使用marker进行OCR文本注入")
    parser.add_argument("--path", required=True, help="需要扫描的包含md文件的目标文件夹路径")
    parser.add_argument("--no-backup", action="store_true", help="关闭自动生成 .bak 备份文件的功能")
    parser.add_argument("--torch-device", default="cuda", help="传给 marker 的 TORCH_DEVICE，默认 cuda，要求使用 GPU 加速")
    parser.add_argument("--remove-images", action="store_true", help="OCR 文字注入成功后删除原始图片文件（仅在 Markdown 安全写入后执行）")
    args = parser.parse_args()

    target_dir = Path(args.path)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[致命错误] 目标路径不存在或不是一个目录: {target_dir}")
        sys.exit(1)

    # 递归遍历所有 .md 文件
    md_files = list(target_dir.rglob("*.md"))
    print(f"=== 开始全盘扫描 ===")
    print(f"目标目录: {target_dir.resolve()}")
    print(f"找到 Markdown 文件共计: {len(md_files)} 个")
    print(f"Marker TORCH_DEVICE: {args.torch_device}")
    print(f"=====================")

    for idx, md_file in enumerate(md_files, 1):
        print(f"\n进度: [{idx}/{len(md_files)}]")
        process_md_file(md_file, skip_backup=args.no_backup, torch_device=args.torch_device, remove_images=args.remove_images)

    print("\n[🎉 运行结束] 所有文档批量 OCR 注入任务处理完毕！")

if __name__ == "__main__":
    main()

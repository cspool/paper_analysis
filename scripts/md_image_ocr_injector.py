import os
import sys
import re
import shutil
import argparse
import tempfile
import subprocess
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

DEFAULT_MARKER_ROOT = Path("/data3/Projects/marker")
DEFAULT_MARKER_PYTHON = Path("/home/descfly/miniconda3/bin/python3")

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


def resolve_local_image(src, md_file_dir):
    """将 Markdown 本地图片路径解析为绝对路径；网络图片返回 None。"""
    normalized = unquote(normalize_markdown_src(src))
    if normalized.startswith(('http://', 'https://')):
        return None
    img_path = Path(normalized)
    if not img_path.is_absolute():
        return (Path(md_file_dir) / img_path).resolve()
    return img_path.resolve()


def meaningful_ocr_text(ocr_text):
    """Marker 仅返回图片标签时不视为 OCR 成功。"""
    text_without_images = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', ocr_text).strip()
    return bool(text_without_images)


def has_existing_ocr_block(content, img_start):
    return OCR_BLOCK_BEFORE_IMAGE_RE.search(content[:img_start]) is not None


def run_marker_ocr(
    img_path_str,
    md_file_dir,
    torch_device="cuda",
    marker_root=DEFAULT_MARKER_ROOT,
    marker_python=DEFAULT_MARKER_PYTHON,
):
    """
    通过指定 Python 从 Marker 源码目录调用 convert_single.py 执行 OCR，
    并提取生成的 Markdown 文本。
    """
    # 1. 解析图片的绝对路径（处理相对路径问题），URL解码处理Notion导出的编码路径
    abs_img_path = resolve_local_image(img_path_str, md_file_dir)
    if abs_img_path is None:
        return None

    if not abs_img_path.exists():
        print(f"    [跳过] 图片文件不存在: {abs_img_path}")
        return None

    # 2. 创建临时目录接收 marker 输出，避免污染工作区
    with tempfile.TemporaryDirectory() as tmp_dir:
        marker_root = Path(marker_root).expanduser().resolve()
        marker_python = Path(marker_python).expanduser().resolve()
        marker_script = marker_root / "convert_single.py"
        if not marker_python.is_file():
            print(f"    [错误] Marker Python 不存在: {marker_python}")
            return None
        if not marker_script.is_file():
            print(f"    [错误] Marker 单文件转换脚本不存在: {marker_script}")
            return None

        # 与 pdf_to_md.py 使用相同的源码目录、解释器和工作目录，避免全局
        # marker_single 入口无法导入本地 marker 包。
        command = [
            str(marker_python),
            str(marker_script),
            str(abs_img_path),
            "--output_dir",
            str(tmp_dir),
            "--disable_multiprocessing",
            # 单张导出图默认会被 Marker 视为 Picture 并再次输出图片标签；
            # 强制 Text block 才会对图内文字执行检测与识别。
            "--force_layout_block",
            "Text",
        ]
        marker_env = os.environ.copy()
        marker_env["TORCH_DEVICE"] = torch_device
        
        try:
            # 设置 120 秒超时防止单张图片长期卡住。
            result = subprocess.run(
                command,
                cwd=marker_root,
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
            # 防止 Marker 仍只返回图片标签时把它误当作 OCR 文本注入原文。
            if not meaningful_ocr_text(ocr_text):
                print("    [警告] Marker 未识别出图内文字，仅返回图片标签，保持原文不变")
                return None
            return ocr_text

        except subprocess.TimeoutExpired:
            print(f"    [超时] 该图片 OCR 处理超时（超过 120 秒），已自动跳过")
            return None
        except Exception as e:
            print(f"    [异常] 执行过程中出现系统错误: {e}")
            return None


def run_marker_ocr_batch(
    image_paths,
    cuda_devices,
    workers_per_gpu=3,
    torch_device="cuda",
    marker_root=DEFAULT_MARKER_ROOT,
    marker_python=DEFAULT_MARKER_PYTHON,
):
    """
    使用一张 GPU 一个 Marker batch 进程批量 OCR 图片。

    每个进程只加载一次模型；主进程等待全部分片结束后返回
    {绝对图片路径字符串: OCR 文本或 None}，不在并发阶段修改 Markdown。
    """
    marker_root = Path(marker_root).expanduser().resolve()
    marker_python = Path(marker_python).expanduser().resolve()
    marker_script = marker_root / "convert.py"
    unique_images = sorted({Path(path).resolve() for path in image_paths}, key=str)
    results = {str(path): None for path in unique_images}
    if not unique_images:
        return results
    if not cuda_devices:
        raise ValueError("cuda_devices 不能为空")
    if not marker_python.is_file():
        raise FileNotFoundError(f"Marker Python 不存在: {marker_python}")
    if not marker_script.is_file():
        raise FileNotFoundError(f"Marker 批量转换脚本不存在: {marker_script}")

    with tempfile.TemporaryDirectory(prefix="marker-image-ocr-batch-") as raw_tmp:
        temp_root = Path(raw_tmp)
        output_root = temp_root / "output"
        output_root.mkdir()
        staging = temp_root / "staging"
        staging.mkdir()
        base_to_image = {}
        for index, image_path in enumerate(unique_images):
            digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
            staged_name = f"image_{index:06d}_{digest}{image_path.suffix.lower()}"
            staged_path = staging / staged_name
            staged_path.symlink_to(image_path)
            base_name = staged_path.stem
            base_to_image[base_name] = image_path

        processes = []
        try:
            for chunk_idx, device in enumerate(cuda_devices):
                stdout_path = temp_root / f"gpu-{device}.stdout.log"
                stderr_path = temp_root / f"gpu-{device}.stderr.log"
                stdout_handle = stdout_path.open("w", encoding="utf-8")
                stderr_handle = stderr_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(device)
                env["TORCH_DEVICE"] = torch_device
                command = [
                    str(marker_python),
                    str(marker_script),
                    str(staging),
                    "--output_dir",
                    str(output_root),
                    "--num_chunks",
                    str(len(cuda_devices)),
                    "--chunk_idx",
                    str(chunk_idx),
                    "--workers",
                    str(workers_per_gpu),
                    "--max_tasks_per_worker",
                    "1000000",
                    "--force_layout_block",
                    "Text",
                ]
                chunk_size = (len(unique_images) + len(cuda_devices) - 1) // len(cuda_devices)
                start = chunk_idx * chunk_size
                assigned_count = max(0, min(len(unique_images), start + chunk_size) - start)
                if assigned_count == 0:
                    stdout_handle.close()
                    stderr_handle.close()
                    continue
                print(
                    f"[批量 OCR] GPU {device}: {assigned_count} 张图片，"
                    f"{workers_per_gpu} 个持久 worker"
                )
                process = subprocess.Popen(
                    command,
                    cwd=marker_root,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
                processes.append(
                    (device, process, stdout_handle, stderr_handle, stderr_path)
                )

            started_at = time.monotonic()
            while any(process.poll() is None for _, process, *_ in processes):
                completed = len(list(output_root.glob("*/*.md")))
                elapsed = format_elapsed(time.monotonic() - started_at)
                device_progress = []
                for device, process, _stdout, _stderr, stderr_path in processes:
                    current, total = read_marker_progress(stderr_path)
                    state = "运行中" if process.poll() is None else "完成"
                    if total:
                        device_progress.append(
                            f"GPU {device} {current}/{total} {state}"
                        )
                    else:
                        device_progress.append(f"GPU {device} {state}")
                percent = completed * 100 / len(unique_images)
                print(
                    f"\r[批量 OCR] 总进度 {completed}/{len(unique_images)} "
                    f"({percent:.1f}%) | 用时 {elapsed} | "
                    + " | ".join(device_progress),
                    end="",
                    flush=True,
                )
                time.sleep(2)
            completed = len(list(output_root.glob("*/*.md")))
            elapsed = format_elapsed(time.monotonic() - started_at)
            print(
                f"\r[批量 OCR] 总进度 {completed}/{len(unique_images)} "
                f"({completed * 100 / len(unique_images):.1f}%) | 用时 {elapsed}"
                + " " * 40
            )

            for device, process, stdout_handle, stderr_handle, stderr_path in processes:
                returncode = process.wait()
                stdout_handle.close()
                stderr_handle.close()
                if returncode != 0:
                    error_tail = "\n".join(
                        stderr_path.read_text(encoding="utf-8", errors="replace")
                        .splitlines()[-20:]
                    )
                    print(f"[批量 OCR 错误] GPU {device} Exit Code {returncode}")
                    if error_tail:
                        print(error_tail)
                else:
                    print(f"[批量 OCR] GPU {device}: 完成")
        except KeyboardInterrupt:
            for _device, process, _stdout, _stderr, _path in processes:
                if process.poll() is None:
                    process.terminate()
            for _device, process, stdout_handle, stderr_handle, _path in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                if not stdout_handle.closed:
                    stdout_handle.close()
                if not stderr_handle.closed:
                    stderr_handle.close()
            raise
        finally:
            for _device, _process, stdout_handle, stderr_handle, _path in processes:
                if not stdout_handle.closed:
                    stdout_handle.close()
                if not stderr_handle.closed:
                    stderr_handle.close()

        for base_name, image_path in base_to_image.items():
            generated = output_root / base_name / f"{base_name}.md"
            if not generated.is_file():
                print(f"    [警告] 未找到 OCR 输出: {image_path}")
                continue
            ocr_text = generated.read_text(encoding="utf-8").strip()
            if not meaningful_ocr_text(ocr_text):
                print(f"    [跳过] 未识别出图内文字: {image_path}")
                continue
            results[str(image_path)] = ocr_text
    return results


def read_marker_progress(stderr_path):
    """从 Marker tqdm 日志尾部提取最后一个 current/total。"""
    try:
        with stderr_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0, 0
    matches = re.findall(r"Processing PDFs:.*?(\d+)/(\d+)", tail)
    if not matches:
        return 0, 0
    current, total = matches[-1]
    return int(current), int(total)


def format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def collect_pending_images(md_files):
    """收集尚未注入 OCR 的本地图片，供多 GPU 批量转换。"""
    pending = set()
    for md_path in md_files:
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[文件读取失败] {md_path}: {e}")
            continue
        for match in iter_markdown_images(content):
            if has_existing_ocr_block(content, match.start):
                continue
            abs_img_path = resolve_local_image(match.src, md_path.parent)
            if abs_img_path is None:
                continue
            if not abs_img_path.exists():
                print(f"    [跳过] 图片文件不存在: {abs_img_path}")
                continue
            pending.add(abs_img_path)
    return sorted(pending, key=str)

def process_md_file(
    md_path,
    skip_backup=False,
    torch_device="cuda",
    remove_images=False,
    marker_root=DEFAULT_MARKER_ROOT,
    marker_python=DEFAULT_MARKER_PYTHON,
    precomputed_ocr=None,
):
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

        abs_img_path = resolve_local_image(src, md_dir)
        if precomputed_ocr is not None:
            ocr_result = precomputed_ocr.get(str(abs_img_path))
        else:
            print(f"  --> 正在 OCR 图片: {src}")
            ocr_result = run_marker_ocr(
                src,
                md_dir,
                torch_device=torch_device,
                marker_root=marker_root,
                marker_python=marker_python,
            )
        
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
    parser.add_argument(
        "--cuda-devices",
        help="启用多 GPU 批量模式，例如 0,1；图片按设备均匀分片",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="多 GPU 批量模式中每张卡的持久 Marker worker 数，默认 3（24GB 4090）",
    )
    parser.add_argument(
        "--marker-root",
        default=os.environ.get("MARKER_ROOT", str(DEFAULT_MARKER_ROOT)),
        help=f"Marker 源码根目录，默认 {DEFAULT_MARKER_ROOT}",
    )
    parser.add_argument(
        "--marker-python",
        default=os.environ.get("MARKER_PYTHON", str(DEFAULT_MARKER_PYTHON)),
        help=f"Marker Python 解释器，默认 {DEFAULT_MARKER_PYTHON}",
    )
    args = parser.parse_args()

    target_dir = Path(args.path)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[致命错误] 目标路径不存在或不是一个目录: {target_dir}")
        sys.exit(1)

    marker_root = Path(args.marker_root).expanduser().resolve()
    marker_python = Path(args.marker_python).expanduser().resolve()
    marker_script = marker_root / "convert_single.py"
    if not marker_root.is_dir():
        print(f"[致命错误] Marker 源码根目录不存在: {marker_root}")
        sys.exit(2)
    if not marker_python.is_file():
        print(f"[致命错误] Marker Python 不存在: {marker_python}")
        sys.exit(2)
    if not marker_script.is_file():
        print(f"[致命错误] Marker 单文件转换脚本不存在: {marker_script}")
        sys.exit(2)

    # 递归遍历所有 .md 文件
    md_files = list(target_dir.rglob("*.md"))
    print(f"=== 开始全盘扫描 ===")
    print(f"目标目录: {target_dir.resolve()}")
    print(f"找到 Markdown 文件共计: {len(md_files)} 个")
    print(f"Marker TORCH_DEVICE: {args.torch_device}")
    print(f"Marker 根目录: {marker_root}")
    print(f"Marker Python: {marker_python}")
    cuda_devices = None
    if args.cuda_devices:
        cuda_devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
        if not cuda_devices:
            print("[致命错误] --cuda-devices 未包含有效设备编号")
            sys.exit(2)
        if any(re.fullmatch(r"\d+", device) is None for device in cuda_devices):
            print("[致命错误] --cuda-devices 只接受逗号分隔的物理 GPU 编号，例如 0,1")
            sys.exit(2)
        if args.torch_device != "cuda":
            print("[致命错误] --cuda-devices 仅可与 --torch-device cuda 一起使用")
            sys.exit(2)
        if args.workers_per_gpu <= 0:
            print("[致命错误] --workers-per-gpu 必须为正整数")
            sys.exit(2)
        print(f"CUDA 物理设备: {', '.join(cuda_devices)}")
        print(f"每张 GPU 的 Marker workers: {args.workers_per_gpu}")
    print(f"=====================")

    try:
        precomputed_ocr = None
        if cuda_devices:
            pending_images = collect_pending_images(md_files)
            print(f"待批量 OCR 的唯一图片: {len(pending_images)}")
            precomputed_ocr = run_marker_ocr_batch(
                pending_images,
                cuda_devices,
                workers_per_gpu=args.workers_per_gpu,
                torch_device=args.torch_device,
                marker_root=marker_root,
                marker_python=marker_python,
            )
        for idx, md_file in enumerate(md_files, 1):
            print(f"\n进度: [{idx}/{len(md_files)}]")
            process_md_file(
                md_file,
                skip_backup=args.no_backup,
                torch_device=args.torch_device,
                remove_images=args.remove_images,
                marker_root=marker_root,
                marker_python=marker_python,
                precomputed_ocr=precomputed_ocr,
            )
    except KeyboardInterrupt:
        print("\n[已中断] 收到 Ctrl-C；已完成写入的 Markdown 保持有效，可重复运行继续。")
        sys.exit(130)

    print("\n[🎉 运行结束] 所有文档批量 OCR 注入任务处理完毕！")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test Obsidian MCP Server API - obsidian_get_note
"""

import subprocess
import json
import time
import sys

def test_obsidian_get_note():
    """测试 obsidian_get_note API"""
    
    print("=" * 60)
    print("Obsidian MCP Server - obsidian_get_note API Test")
    print("=" * 60)
    
    # 启动 MCP 服务器
    print("\n[1] 启动 MCP 服务器...")
    env = {
        'MCP_TRANSPORT_TYPE': 'stdio',
        'MCP_LOG_LEVEL': 'error',
        'OBSIDIAN_API_KEY': '6fd0885f6fd8a8ac5b95fd2885988815cd733b61b9d2343009edfa18dfbbdfc8',
        'MCP_ENABLE_COMMANDS': 'true'
    }
    
    try:
        proc = subprocess.Popen(
            ['bunx', 'obsidian-mcp-server@latest'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**subprocess.os.environ, **env}
        )
        time.sleep(2)
        print("✓ MCP 服务器已启动")
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        return
    
    # 第一步：初始化
    print("\n[2] 发送 initialize 请求...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    try:
        proc.stdin.write(json.dumps(init_request) + '\n')
        proc.stdin.flush()
        
        # 读取初始化响应
        response_line = proc.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            print(f"✓ 初始化响应: {json.dumps(response, ensure_ascii=False)[:200]}...")
        else:
            print("✗ 没有收到响应")
            proc.terminate()
            return
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        proc.terminate()
        return
    
    # 第二步：列出可用工具
    print("\n[3] 发送 tools/list 请求...")
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    try:
        proc.stdin.write(json.dumps(tools_request) + '\n')
        proc.stdin.flush()
        
        response_line = proc.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                print(f"✓ 找到 {len(tools)} 个工具:")
                for tool in tools[:5]:
                    print(f"  - {tool['name']}")
                if len(tools) > 5:
                    print(f"  ... 以及 {len(tools) - 5} 个其他工具")
            else:
                print(f"✗ 响应格式错误: {response}")
        else:
            print("✗ 没有收到响应")
            proc.terminate()
            return
    except Exception as e:
        print(f"✗ 列表请求失败: {e}")
        proc.terminate()
        return
    
    # 第三步：调用 obsidian_get_note
    print("\n[4] 发送 obsidian_get_note 请求...")
    get_note_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "obsidian_get_note",
            "arguments": {
                "target": {
                    "type": "path",
                    "path": "knowledge_notes/系统知识笔记/Modal Cache（模态缓存）.md"
                },
                "format": "content"
            }
        }
    }
    
    try:
        proc.stdin.write(json.dumps(get_note_request) + '\n')
        proc.stdin.flush()
        
        response_line = proc.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            print(f"\n✓ 获取笔记成功!")
            
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"]
                preview = content[:300] if isinstance(content, str) else str(content)[:300]
                print(f"\n笔记内容预览:")
                print("-" * 60)
                print(preview)
                print("-" * 60)
                print(f"\n（共 {len(content) if isinstance(content, str) else len(str(content))} 字符）")
            else:
                print(f"\n完整响应:")
                print(json.dumps(response, ensure_ascii=False, indent=2)[:500])
        else:
            print("✗ 没有收到响应")
    except json.JSONDecodeError as e:
        print(f"✗ JSON 解析失败: {e}")
        # 尝试读取错误信息
        try:
            error_line = proc.stderr.readline()
            if error_line:
                print(f"  错误日志: {error_line}")
        except:
            pass
    except Exception as e:
        print(f"✗ 请求失败: {e}")
    finally:
        print("\n[5] 关闭连接...")
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("✓ 测试完成")

if __name__ == '__main__':
    test_obsidian_get_note()

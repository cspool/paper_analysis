#!/usr/bin/env python3
"""
Test Obsidian MCP Server - obsidian_search_notes (omnisearch mode)
"""

import subprocess
import json
import time

def test_obsidian_search():
    """测试 obsidian_search_notes omnisearch API"""
    
    print("=" * 60)
    print("Obsidian MCP Server - obsidian_search_notes Test")
    print("=" * 60)
    
    # 启动 MCP 服务器
    print("\n[1] 启动 MCP 服务器...")
    env = {
        'MCP_TRANSPORT_TYPE': 'stdio',
        'MCP_LOG_LEVEL': 'error',
        'OBSIDIAN_API_KEY': '6fd0885f6fd8a8ac5b95fd2885988815cd733b61b9d2343009edfa18dfbbdfc8',
        'MCP_ENABLE_COMMANDS': 'true'
    }
    
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
    
    # 初始化
    print("\n[2] 初始化...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    proc.stdin.write(json.dumps(init_request) + '\n')
    proc.stdin.flush()
    response = json.loads(proc.stdout.readline())
    print(f"✓ 初始化完成")
    
    # 搜索测试
    print("\n[3] 测试 omnisearch 搜索...")
    search_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "obsidian_search_notes",
            "arguments": {
                "mode": "omnisearch",
                "query": "KV Cache"
            }
        }
    }
    
    try:
        proc.stdin.write(json.dumps(search_request) + '\n')
        proc.stdin.flush()
        
        response_line = proc.stdout.readline()
        response = json.loads(response_line)
        
        if "result" in response and "results" in response["result"]:
            results = response["result"]["results"]
            print(f"\n✓ 搜索成功! 找到 {len(results)} 个结果")
            print(f"  总数 (totalCount): {response['result'].get('totalCount', '?')}")
            print(f"  是否被截断: {response['result'].get('truncated', False)}")
            
            print("\n搜索结果预览:")
            print("-" * 60)
            for i, result in enumerate(results[:3], 1):
                print(f"\n{i}. {result.get('path', 'unknown')}")
                print(f"   分数: {result.get('score', 'N/A')}")
                snippet = result.get('snippet', '')[:150]
                if snippet:
                    print(f"   摘要: {snippet}...")
            if len(results) > 3:
                print(f"\n   ... 以及 {len(results) - 3} 个其他结果")
        else:
            print(f"✗ 响应格式错误:")
            print(json.dumps(response, ensure_ascii=False, indent=2)[:500])
    except Exception as e:
        print(f"✗ 搜索失败: {e}")
    finally:
        print("\n[4] 关闭连接...")
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("✓ 测试完成")

if __name__ == '__main__':
    test_obsidian_search()

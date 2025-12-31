import requests
import json
import pandas as pd
import time
import re
from datetime import datetime
from tqdm import tqdm
import urllib.parse


class BiliBiliCommentCrawlerUltimate:
    def __init__(self, cookies_str=None):
        """终极版B站评论爬虫"""
        self.session = requests.Session()
        
        # 最新版请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Origin': 'https://www.bilibili.com',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
        }
        
        if cookies_str:
            self._set_cookies(cookies_str)
        
        self.session.headers.update(self.headers)
    
    def _set_cookies(self, cookies_str):
        """设置cookies"""
        cookies_dict = {}
        for item in cookies_str.split(';'):
            item = item.strip()
            if '=' in item:
                key, value = item.split('=', 1)
                cookies_dict[key] = value
                self.session.cookies.set(key, value)
        print(f"已设置Cookies: {list(cookies_dict.keys())}")
    
    def debug_api_structure(self, bvid):
        """
        调试函数：完整分析API结构
        """
        print("=" * 70)
        print("🔍 API结构深度调试")
        print("=" * 70)
        
        # 1. 获取视频基本信息
        print("\n1. 获取视频基本信息...")
        video_info = self._get_video_info_raw(bvid)
        if not video_info:
            print("❌ 无法获取视频信息")
            return
        
        aid = video_info.get('aid')
        print(f"✅ 视频aid: {aid}")
        print(f"   标题: {video_info.get('title', '')[:50]}...")
        print(f"   评论数: {video_info.get('reply', 0)}")
        
        # 2. 测试不同API端点
        print("\n2. 测试不同API端点...")
        
        api_endpoints = [
            # 最新版API（推荐）
            {
                'name': '新版主API',
                'url': f'https://api.bilibili.com/x/v2/reply/main?type=1&oid={aid}&next=0',
                'method': 'GET'
            },
            # 传统API（备选）
            {
                'name': '传统分页API',
                'url': f'https://api.bilibili.com/x/v2/reply?type=1&oid={aid}&pn=1&ps=20&sort=2',
                'method': 'GET'
            },
            # 网页版API
            {
                'name': '网页版API',
                'url': f'https://api.bilibili.com/x/v2/reply/wbi/main?type=1&oid={aid}&next=1',
                'method': 'GET'
            }
        ]
        
        for endpoint in api_endpoints:
            print(f"\n测试 {endpoint['name']}:")
            print(f"URL: {endpoint['url']}")
            
            try:
                response = self.session.get(endpoint['url'], timeout=10)
                print(f"状态码: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"返回code: {data.get('code')}")
                    print(f"返回message: {data.get('message', 'N/A')}")
                    
                    # 分析数据结构
                    if 'data' in data:
                        data_keys = list(data['data'].keys())
                        print(f"data字段包含: {data_keys}")
                        
                        # 检查是否有评论数据
                        if 'replies' in data['data']:
                            replies = data['data']['replies']
                            print(f"找到replies字段，类型: {type(replies)}，数量: {len(replies) if isinstance(replies, list) else 'N/A'}")
                            if replies and len(replies) > 0:
                                print(f"✅ 成功获取到评论！第一条评论预览: {replies[0].get('content', {}).get('message', '')[:50]}...")
                        
                        if 'cursor' in data['data']:
                            cursor = data['data']['cursor']
                            print(f"cursor字段: {cursor}")
                    
                    # 显示部分响应内容（调试用）
                    print(f"响应摘要: {json.dumps(data, ensure_ascii=False)[:300]}...")
                    
                else:
                    print(f"❌ 请求失败: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 测试出错: {e}")
            
            time.sleep(1)
        
        # 3. 测试真实爬取
        print("\n" + "=" * 70)
        print("3. 测试真实爬取少量数据...")
        
        test_comments = self._test_crawl_comments(aid, max_pages=2)
        print(f"测试爬取结果: 获取到 {len(test_comments)} 条评论")
        
        if test_comments:
            print(f"示例评论:")
            for i, comment in enumerate(test_comments[:3], 1):
                print(f"{i}. [{comment.get('user_name', '')}] {comment.get('comment_text', '')[:60]}...")
        
        print("\n" + "=" * 70)
        print("调试完成！")
    
    def _get_video_info_raw(self, bvid):
        """获取视频原始信息"""
        url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 0:
                    return data['data']
        except:
            pass
        return None
    
    def _test_crawl_comments(self, aid, max_pages=2):
        """测试爬取少量评论"""
        comments = []
        
        # 使用新版API
        next_param = 0
        
        for page in range(1, max_pages + 1):
            try:
                # 使用新版main API
                url = f'https://api.bilibili.com/x/v2/reply/main?type=1&oid={aid}&next={next_param}'
                
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('code') == 0 and 'data' in data:
                        # 提取评论
                        replies = data['data'].get('replies', [])
                        if replies:
                            for reply in replies:
                                comment = self._parse_reply(reply)
                                if comment:
                                    comments.append(comment)
                        
                        # 更新next参数
                        if 'cursor' in data['data']:
                            next_param = data['data']['cursor'].get('next', page)
                        else:
                            next_param = page
                    
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"测试爬取第{page}页出错: {e}")
                break
        
        return comments
    
    def _parse_reply(self, reply):
        """解析评论回复"""
        try:
            return {
                'comment_id': reply.get('rpid'),
                'user_id': reply.get('member', {}).get('mid'),
                'user_name': reply.get('member', {}).get('uname'),
                'comment_text': reply.get('content', {}).get('message', ''),
                'like_count': reply.get('like', 0),
                'reply_count': reply.get('count', 0),
                'comment_time': datetime.fromtimestamp(reply.get('ctime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'user_level': reply.get('member', {}).get('level_info', {}).get('current_level', 0),
            }
        except:
            return None
    
    def get_comments_ultimate(self, bvid, max_comments=500):
        """
        终极版评论爬取方法
        使用最新的API和正确的参数
        """
        print(f"\n🚀 开始终极版评论爬取: {bvid}")
        
        # 1. 获取视频信息
        video_info = self._get_video_info_raw(bvid)
        if not video_info:
            print("❌ 无法获取视频信息")
            return []
        
        aid = video_info.get('aid')
        total_comments = video_info.get('stat', {}).get('reply', 0)
        
        print(f"📊 视频信息:")
        print(f"   标题: {video_info.get('title', '')[:60]}...")
        print(f"   AV号: {aid}")
        print(f"   评论总数: {total_comments}")
        print(f"   目标爬取: {max_comments} 条")
        
        # 2. 使用新版API爬取
        all_comments = []
        next_param = 1  # 新版API使用next参数而不是pn
        page_count = 0
        
        with tqdm(total=min(max_comments, total_comments), 
                 desc="爬取进度", unit="条", ncols=70) as pbar:
            
            while len(all_comments) < max_comments:
                page_count += 1
                
                try:
                    # 关键：使用新版main API
                    url = f'https://api.bilibili.com/x/v2/reply/main?type=1&oid={aid}&next={next_param}&mode=3'
                    
                    print(f"\n📄 请求第{page_count}页, next={next_param}")
                    print(f"   URL: {url}")
                    
                    response = self.session.get(url, timeout=15)
                    print(f"   状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # 调试：显示API返回结构
                        if page_count == 1:
                            print(f"   API返回code: {data.get('code')}")
                            if 'data' in data:
                                print(f"   data字段包含: {list(data['data'].keys())}")
                        
                        if data.get('code') == 0 and 'data' in data:
                            # 提取评论（可能有多个位置）
                            replies = []
                            
                            # 位置1：data.replies
                            if 'replies' in data['data'] and data['data']['replies']:
                                replies.extend(data['data']['replies'])
                            
                            # 位置2：data.replies.replies（嵌套回复）
                            # 位置3：data.top.replies（置顶评论）
                            if 'top' in data['data'] and data['data']['top']:
                                top_replies = data['data']['top'].get('replies', [])
                                if top_replies:
                                    replies.extend(top_replies)
                            
                            print(f"   找到 {len(replies)} 条评论")
                            
                            if replies:
                                for reply in replies:
                                    comment = self._parse_comment_ultimate(reply, aid, bvid)
                                    if comment:
                                        all_comments.append(comment)
                                
                                # 更新进度条
                                pbar.update(len(replies))
                                
                                # 更新next参数
                                if 'cursor' in data['data']:
                                    next_param = data['data']['cursor'].get('next', next_param + 1)
                                    is_end = data['data']['cursor'].get('is_end', False)
                                    
                                    print(f"   下一页next参数: {next_param}, 是否结束: {is_end}")
                                    
                                    if is_end or next_param == 0:
                                        print("   ⚠️ 已到评论末尾")
                                        break
                                else:
                                    # 传统分页方式
                                    next_param += 1
                                    if len(replies) < 20:
                                        print("   ⚠️ 评论数不足20条，可能已到末尾")
                                        break
                            
                            else:
                                print("   ⚠️ 本页无评论数据")
                                break
                            
                        else:
                            print(f"   ❌ API错误: code={data.get('code')}, message={data.get('message')}")
                            break
                        
                        # 礼貌延时
                        time.sleep(1)
                        
                    else:
                        print(f"   ❌ 请求失败: HTTP {response.status_code}")
                        break
                        
                except Exception as e:
                    print(f"   ❌ 爬取第{page_count}页出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(3)
                    break
        
        print(f"\n✅ 爬取完成！共获取 {len(all_comments)} 条评论")
        
        # 显示统计信息
        if all_comments:
            self._show_detailed_stats(all_comments)
        
        return all_comments
    
    def _parse_comment_ultimate(self, reply, aid, bvid):
        """终极版评论解析"""
        try:
            # 基础信息
            member = reply.get('member', {})
            content = reply.get('content', {})
            
            comment_data = {
                'comment_id': reply.get('rpid'),
                'video_aid': aid,
                'video_bvid': bvid,
                'user_id': member.get('mid'),
                'user_name': member.get('uname'),
                'user_level': member.get('level_info', {}).get('current_level', 0),
                'is_vip': 1 if member.get('vip', {}).get('status') == 1 else 0,
                'comment_text': content.get('message', ''),
                'comment_text_clean': self._clean_text(content.get('message', '')),
                'like_count': reply.get('like', 0),
                'reply_count': reply.get('count', 0),
                'comment_time': datetime.fromtimestamp(reply.get('ctime', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_gender': member.get('sex', '未知'),
                'user_avatar': member.get('avatar', ''),
            }
            
            return comment_data
            
        except Exception as e:
            print(f"解析评论出错: {e}")
            return None
    
    def _clean_text(self, text):
        """清洗文本"""
        if not text:
            return ""
        # 移除特殊字符但保留中文和基本标点
        text = re.sub(r'http[s]?://\S+', '', text)  # 移除URL
        text = re.sub(r'@\S+', '', text)  # 移除@
        text = re.sub(r'\[.*?\]', '', text)  # 移除表情
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _show_detailed_stats(self, comments):
        """显示详细统计"""
        print("\n" + "=" * 60)
        print("📈 详细统计信息")
        print("=" * 60)
        
        print(f"总评论数: {len(comments)}")
        
        # 点赞分析
        likes = [c['like_count'] for c in comments]
        if likes:
            avg_likes = sum(likes) / len(likes)
            max_likes = max(likes)
            print(f"平均点赞: {avg_likes:.1f}")
            print(f"最高点赞: {max_likes}")
        
        # 用户分析
        unique_users = len(set(c['user_id'] for c in comments if c['user_id']))
        print(f"独立用户数: {unique_users}")
        
        # VIP分析
        vip_count = sum(1 for c in comments if c['is_vip'] == 1)
        if comments:
            print(f"VIP用户占比: {vip_count/len(comments)*100:.1f}%")
        
        # 时间分析
        if comments:
            times = [c['comment_time'] for c in comments if c['comment_time']]
            if times:
                print(f"最早评论: {min(times)}")
                print(f"最新评论: {max(times)}")
        
        # 热门评论
        if len(comments) >= 5:
            top_comments = sorted(comments, key=lambda x: x['like_count'], reverse=True)[:5]
            print(f"\n🔥 热门评论Top 5:")
            for i, comment in enumerate(top_comments, 1):
                preview = comment['comment_text_clean'][:40] + "..." if len(comment['comment_text_clean']) > 40 else comment['comment_text_clean']
                print(f"{i}. [{comment['user_name']}] {preview} (👍{comment['like_count']})")
        
        print("=" * 60)
    
    def save_comments_smart(self, comments, topic="新国标"):
        """智能保存评论"""
        if not comments:
            print("❌ 无评论数据可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 保存为Excel
            df = pd.DataFrame(comments)
            excel_file = f"bilibili_{topic}_comments_{timestamp}.xlsx"
            df.to_excel(excel_file, index=False, engine='openpyxl')
            print(f"✅ Excel文件: {excel_file}")
            
            # 保存为CSV备份
            csv_file = f"bilibili_{topic}_comments_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ CSV备份文件: {csv_file}")
            
            # 保存纯文本（用于词云）
            text_file = f"bilibili_{topic}_text_{timestamp}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for comment in comments:
                    if comment['comment_text_clean'].strip():
                        f.write(comment['comment_text_clean'] + "\n")
            print(f"✅ 纯文本文件: {text_file} ({len(comments)}条评论)")
            
            return excel_file
            
        except Exception as e:
            print(f"❌ 保存文件时出错: {e}")
            return None


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 B站评论爬虫 - 终极修复版")
    print("=" * 70)
    
    # 初始化
    cookies = "SESSDATA=27eb8e53%2C1782662548%2C677ef%2Ac1CjBpLirhYHLYZBkDWL6mPadJURzavymVdhiW8s-IViafEeMaeudmAsDBV3R5cIVZPoESVmZGZVVtSnZNQUQxUU1IRkZuOVFwNHJ6Y1ZPcDFpS1BvdHVjS0htbjRFQUtHYm5tTm1Pa2g1ZTJkRDcxM0Nyc1ptdHFfSlV3bURTMXgwTGlpbUkwUWFBIIEC;bili_jct=656b2c64cbb26b795f6d124b0a9ac8f5;DedeUserID=3546712461281929"  # 替换为实际Cookies
    crawler = BiliBiliCommentCrawlerUltimate(cookies if cookies != "你的B站Cookies" else None)
    
    # 目标视频（电动车新国标相关）
    target_videos = [
        "BV1Q5mbBnEmQ",  # 你知道新国标改了什么吗你就喷？
        "BV1UJ4m1Y7Kc",  # 新国标电动车实测
        "BV1bZ42177Dp",  # 央视报道新国标
    ]
    
    all_comments = []
    
    for bvid in target_videos:
        print(f"\n{'='*70}")
        print(f"处理视频: {bvid}")
        print(f"{'='*70}")
        
        # 可选：先调试API结构
        # crawler.debug_api_structure(bvid)
        
        # 爬取评论
        comments = crawler.get_comments_ultimate(bvid, max_comments=300)
        
        # 添加到总列表
        all_comments.extend(comments)
        
        # 保存当前视频评论
        if comments:
            crawler.save_comments_smart(comments, f"新国标_{bvid}")
        
        # 延时
        if bvid != target_videos[-1]:
            print("等待3秒处理下一个视频...")
            time.sleep(3)
    
    # 保存所有评论
    if all_comments:
        print(f"\n{'='*70}")
        print(f"✅ 所有视频爬取完成！")
        print(f"📊 总计评论数: {len(all_comments)}")
        
        final_file = crawler.save_comments_smart(all_comments, "新国标_汇总")
        
        if final_file:
            print(f"\n🎉 数据收集完成！")
            print(f"📁 主要文件: {final_file}")
            print(f"\n下一步建议:")
            print(f"1. 使用文本文件生成词云")
            print(f"2. 进行情感分析和主题提取")
            print(f"3. 结合Excel数据进行深度分析")


if __name__ == "__main__":
    main()
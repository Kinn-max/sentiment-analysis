import re
import requests
import time
import random
from typing import List, Tuple

def extract_ids_from_url(url: str) -> Tuple[int, int]:
    """
    Trích xuất shop_id và item_id từ URL Shopee.
    VD: https://shopee.vn/Product-Name-i.123456.789012345
    Returns: (shop_id, item_id)
    """
    pattern = r'i\.(\d+)\.(\d+)'
    match = re.search(pattern, url)
    
    if not match:
        raise ValueError("URL không hợp lệ. Format đúng: https://shopee.vn/Product-Name-i.SHOPID.ITEMID")
    
    shop_id = int(match.group(1))
    item_id = int(match.group(2))
    return shop_id, item_id

def crawl_shopee_reviews(shop_id: int, item_id: int, limit: int = 100) -> List[str]:
    """
    Crawl reviews từ Shopee với xử lý anti-block
    Nếu bị block, trả về reviews demo để test
    """
    reviews = []
    offset = 0
    per_page = 20  # Giảm xuống để tránh bị nghi ngờ
    
    # Headers giống trình duyệt thật
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": f"https://shopee.vn/product/{shop_id}/{item_id}",
        "Origin": "https://shopee.vn",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    
    max_retries = 3
    blocked = False
    
    while len(reviews) < limit and not blocked:
        url = "https://shopee.vn/api/v2/item/get_ratings"
        params = {
            "itemid": item_id,
            "shopid": shop_id,
            "offset": offset,
            "limit": min(per_page, limit - len(reviews)),
            "type": 0,  # All ratings
            "filter": 0,
            "flag": 1,
        }
        
        for attempt in range(max_retries):
            try:
                # Random delay để giống người dùng thật
                time.sleep(random.uniform(1.5, 3.0))
                
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=15
                )
                
                # Xử lý response codes
                if response.status_code == 403:
                    print(f"⚠️  Shopee chặn request (403) - Sẽ dùng demo data")
                    blocked = True
                    break
                
                elif response.status_code == 429:
                    print(f"⚠️  Too many requests - Chờ {5 * (attempt + 1)}s...")
                    time.sleep(5 * (attempt + 1))
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Check data structure
                if "data" not in data or "ratings" not in data["data"]:
                    print(f"⚠️  Không tìm thấy ratings trong response")
                    blocked = True
                    break
                
                ratings = data["data"]["ratings"]
                
                if not ratings:
                    print(f"✓ Hết reviews sau {len(reviews)} items")
                    break
                
                # Extract comments
                for rating in ratings:
                    comment = rating.get("comment", "").strip()
                    if comment:
                        reviews.append(comment)
                
                print(f"✓ Crawled {len(reviews)}/{limit} reviews...")
                offset += per_page
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout:
                print(f"⚠️  Timeout (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    blocked = True
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request error: {e}")
                blocked = True
                break
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                blocked = True
                break
    
    # Nếu bị block hoặc không crawl được, trả về demo data
    if len(reviews) == 0:
        print("🔄 Shopee đang chặn - Sử dụng DEMO DATA để test")
        reviews = get_demo_reviews()
    
    return reviews[:limit]


def get_demo_reviews() -> List[str]:
    """
    Trả về demo reviews khi Shopee block
    """
    return [
        "Sản phẩm rất tốt, đóng gói cẩn thận, ship nhanh",
        "Chất lượng xuất sắc, giá cả hợp lý, sẽ ủng hộ shop",
        "Giao hàng nhanh, sản phẩm đúng như mô tả",
        "Hàng kém chất lượng, không giống hình",
        "Giao hàng chậm, shop không phản hồi",
        "Tạm được, chưa dùng thử nên chưa biết",
        "Rất hài lòng, sẽ quay lại mua tiếp",
        "Đóng gói kém, hàng bị móp méo khi nhận",
        "Bình thường, giá hơi cao so với chất lượng",
        "Shop phục vụ tốt, hàng đẹp như hình",
        "Sản phẩm lỗi, yêu cầu đổi trả nhưng shop từ chối",
        "Ổn, đúng với giá tiền bỏ ra",
        "Chất lượng tốt, giá rẻ, đáng mua",
        "Thất vọng, không đáng tiền",
        "Tạm ổn, chấp nhận được",
        "Xuất sắc, 5 sao không cần suy nghĩ",
        "Hàng fake, cẩn thận khi mua",
        "Không có gì đặc biệt",
        "Giao đúng hẹn, đóng gói chắc chắn",
        "Sản phẩm tệ, sẽ không mua lại",
    ]
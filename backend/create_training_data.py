
"""
Script để tạo dataset training cho sentiment analysis
Chạy: python create_training_data.py
"""

import pandas as pd
import os

def create_sample_dataset():
    """Tạo dataset mẫu với reviews tiếng Việt thực tế"""
    
    # Dataset mẫu - thay bằng data thực nếu có
    data = {
        'review': [
            # POSITIVE REVIEWS (30 mẫu)
            'Sản phẩm rất tốt, đóng gói cẩn thận, ship nhanh',
            'Chất lượng xuất sắc, giá cả hợp lý, sẽ ủng hộ shop tiếp',
            'Giao hàng nhanh, sản phẩm đúng như mô tả',
            'Rất hài lòng với chất lượng, đáng tiền',
            'Shop phục vụ tốt, hàng đẹp, giống hình',
            'Tuyệt vời, vượt cả mong đợi',
            'Chất lượng tốt, giá rẻ so với thị trường',
            'Đóng gói cẩn thận, hàng ngon',
            'Sản phẩm chất lượng, shop nhiệt tình',
            'Hài lòng lắm, sẽ quay lại',
            'Giá tốt, chất lượng ổn, đáng mua',
            'Ship nhanh, hàng đẹp như hình',
            'Chất lượng ok, giá hợp lý',
            'Sản phẩm tốt, giao đúng hẹn',
            'Đóng gói kỹ càng, hàng đẹp',
            'Rất ưng, sẽ giới thiệu bạn bè',
            'Hàng chất lượng, ship nhanh',
            'Giá rẻ mà chất lượng tốt',
            'Shop uy tín, hàng đẹp',
            'Sản phẩm như mong đợi',
            'Chất lượng tuyệt vời',
            'Đóng gói chắc chắn, hài lòng',
            'Giá tốt, chất lượng ok',
            'Sản phẩm đẹp, giao đúng hẹn',
            'Rất hài lòng, sẽ mua lại',
            'Chất lượng tốt lắm, giá hợp lý',
            'Shop nhiệt tình, hàng đẹp',
            'Giao nhanh, đóng gói cẩn thận',
            'Sản phẩm ổn, đáng tiền',
            'Chất lượng xuất sắc, rất hài lòng',
            
            # NEGATIVE REVIEWS (30 mẫu)
            'Hàng kém chất lượng, không giống hình',
            'Giao hàng chậm, shop không liên lạc được',
            'Sản phẩm lỗi, yêu cầu đổi trả nhưng shop không phản hồi',
            'Đóng gói kém, hàng bị móp méo',
            'Chất lượng tệ, không đáng tiền',
            'Hàng fake, không như mô tả',
            'Thất vọng, sẽ không mua lại',
            'Giao sai màu, shop thái độ không tốt',
            'Hàng lỗi nhưng không được đổi trả',
            'Chất lượng kém, giá lại cao',
            'Không giống hình, rất thất vọng',
            'Giao hàng lâu, đóng gói kém',
            'Sản phẩm kém chất lượng',
            'Shop lừa đảo, hàng fake',
            'Không đáng tiền, rất tệ',
            'Giao sai hàng, shop không giải quyết',
            'Chất lượng tệ hại',
            'Hàng bị lỗi, shop không nhận trách nhiệm',
            'Đóng gói kém, hàng bị vỡ',
            'Không như quảng cáo, lừa khách',
            'Giao chậm, shop thờ ơ',
            'Chất lượng kém, sẽ trả lại',
            'Sản phẩm tệ, không mua nữa',
            'Hàng không đúng mô tả',
            'Shop lừa đảo, cẩn thận',
            'Chất lượng quá tệ',
            'Giao sai size, không đổi được',
            'Hàng kém, giá lại cao',
            'Thất vọng về chất lượng',
            'Shop không uy tín, hàng fake',
            
            # NEUTRAL REVIEWS (30 mẫu)
            'Tạm được, chưa dùng thử',
            'Bình thường, không có gì đặc biệt',
            'Ổn, đúng giá tiền',
            'Tạm ổn, chấp nhận được',
            'Không tốt lắm nhưng cũng không tệ',
            'Bình thường thôi',
            'Tạm được, giá hơi cao',
            'Ổn, chưa test kỹ',
            'Bình thường, không nổi bật',
            'Tạm ổn, đúng mô tả',
            'Không có gì đặc sắc',
            'Ổn áp, giá hợp lý',
            'Tạm được, chờ dùng xem',
            'Bình thường, chấp nhận được',
            'Ổn, không xuất sắc',
            'Tạm ổn thôi',
            'Không tốt lắm',
            'Ổn, đúng như mong đợi',
            'Bình thường',
            'Tạm được',
            'Ổn áp',
            'Không đặc biệt',
            'Tạm chấp nhận',
            'Bình thường thôi, ổn',
            'Tạm được, giá hợp lý',
            'Ổn, chưa dùng nhiều',
            'Không tệ nhưng cũng không tốt',
            'Bình thường, đúng giá',
            'Tạm ổn, sẽ dùng thử',
            'Ổn, không quá ấn tượng',
        ],
        'label': (
            ['positive'] * 30 +
            ['negative'] * 30 +
            ['neutral'] * 30
        )
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle để tránh model học theo thứ tự
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def create_larger_dataset():
    """
    Tạo dataset lớn hơn bằng cách augment data
    Thực tế nên crawl hoặc dùng dataset có sẵn
    """
    import random
    
    # Templates để tạo variations
    positive_templates = [
        "Sản phẩm {adj}, {action}",
        "{adj} lắm, sẽ {future}",
        "Rất {adj}, {comment}",
        "{action}, {adj}"
    ]
    
    positive_words = {
        'adj': ['tốt', 'đẹp', 'chất lượng', 'xuất sắc', 'ổn', 'ngon'],
        'action': ['giao nhanh', 'đóng gói cẩn thận', 'ship nhanh', 'phục vụ tốt'],
        'future': ['mua lại', 'ủng hộ shop', 'giới thiệu bạn bè', 'quay lại'],
        'comment': ['đáng tiền', 'hài lòng', 'như mong đợi', 'giá tốt']
    }
    
    negative_templates = [
        "Hàng {adj}, {problem}",
        "{adj}, sẽ {action}",
        "Rất {adj}, {complaint}"
    ]
    
    negative_words = {
        'adj': ['kém', 'tệ', 'kém chất lượng', 'lỗi', 'fake'],
        'problem': ['không giống hình', 'giao chậm', 'đóng gói kém'],
        'action': ['trả lại', 'không mua nữa', 'báo cáo'],
        'complaint': ['thất vọng', 'không đáng tiền', 'tệ hại']
    }
    
    # Tạo data (simplified - thực tế cần nhiều hơn)
    reviews = []
    labels = []
    
    for _ in range(200):
        template = random.choice(positive_templates)
        review = template
        for key, values in positive_words.items():
            if '{' + key + '}' in review:
                review = review.replace('{' + key + '}', random.choice(values))
        reviews.append(review)
        labels.append('positive')
    
    for _ in range(200):
        template = random.choice(negative_templates)
        review = template
        for key, values in negative_words.items():
            if '{' + key + '}' in review:
                review = review.replace('{' + key + '}', random.choice(values))
        reviews.append(review)
        labels.append('negative')
    
    df = pd.DataFrame({'review': reviews, 'label': labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

if __name__ == "__main__":
    # Tạo thư mục data nếu chưa có
    os.makedirs('data', exist_ok=True)
    
    print("🔄 Đang tạo dataset training...")
    
    # Tạo dataset mẫu (90 reviews)
    df = create_sample_dataset()
    
    # Hoặc tạo dataset lớn hơn (400 reviews - nhưng chất lượng thấp hơn)
    # df = create_larger_dataset()
    
    # Lưu file
    output_path = 'data/reviews.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Đã tạo dataset với {len(df)} reviews")
    print(f"📁 Lưu tại: {output_path}")
    print(f"\n📊 Thống kê:")
    print(df['label'].value_counts())
    print(f"\n🔍 Preview:")
    print(df.head(10))
    
    print("\n💡 Lưu ý:")
    print("- Dataset này CHỈ là mẫu demo")
    print("- Để model chính xác, cần 1000-5000+ reviews thực")
    print("- Tìm dataset tiếng Việt tại:")
    print("  • https://github.com/undertheseanlp/vietnamese-sentiment")
    print("  • https://github.com/sonvx/vietnamese-sentiment-analysis")
    print("  • Hoặc crawl từ Shopee/Tiki/Lazada")
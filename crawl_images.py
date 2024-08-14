import requests
from bs4 import BeautifulSoup
import os

def download_images_from_page(url, folder):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    

    img_tags = soup.find_all('img', {'src': True})
    

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for img in img_tags:
        img_url = img['src']
        if not img_url.startswith('http'):
            img_url = 'https:' + img_url
        
        img_url = img_url.split('?')[0]
        img_name = os.path.join(folder, img_url.split('/')[-1])
        
        try:
            with open(img_name, 'wb') as f:
                f.write(requests.get(img_url).content)
            print(f"Tải xong {img_name}")
        except Exception as e:
            print(f"Không thể tải {img_url}. Lỗi: {e}")

def main():
    url = 'https://hoanghamobile.com/tin-tuc/anh-cho-hai/?srsltid=AfmBOopzj-F2RYvNlUKlKiy5m2prEaP3L3xAt-3uTGKhuMUHRxsG01jA'
    #url_cat = 'https://mytour.vn/vi/blog/bai-viet/top-100-hinh-anh-meo-ngau-nhat.html'
    folder = 'dog_images' #'cat_images'
    
    print(f"Tải ảnh từ {url}")
    download_images_from_page(url, folder)

if __name__ == '__main__':
    main()

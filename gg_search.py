from googleapiclient.discovery import build
from numpy import random
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import re
import logging
import warnings
from typing import List, Tuple, Any
import urllib3
from functools import lru_cache

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore', message='file_cache is unavailable when using oauth2client >= 4.0.0 or google-auth')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Cấu hình hệ thống ghi log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Các hằng số và cấu hình
API_KEYS = [
    'AIzaSyCibUHBhwCBI0XLjrSyTtfm3ewnkrP0LQY', 
    'AIzaSyA_xmw7eBrt9rV0eul-KN9xPEhDar9Le3k',
    'AIzaSyDo9URIA0tEVOgVlXOsivf9PhQTiwl6GqI'
]  # Danh sách các API key để luân phiên sử dụng
CUSTOM_SEARCH_ENGINE_ID = "11de06823e76b4b32"  # ID của Custom Search Engine

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
]  # Danh sách User-Agent để luân phiên sử dụng, tránh bị chặn

# Danh sách các domain tin tức Việt Nam và Wikipedia được phép truy cập
ALLOWED_DOMAINS = [
    # Wikipedia
    'vi.wikipedia.org',
    
    # Các trang báo chính thống
    'vnexpress.net',
    'tuoitre.vn',
    'thanhnien.vn',
    'vietnamnet.vn',
    'dantri.com.vn',
    'nhandan.vn',
    'tienphong.vn',
    'baodautu.vn',
    'cafef.vn',
    'zingnews.vn',
    'vneconomy.vn',
    'nguoiduatin.vn',
    'kenh14.vn',
    'vov.vn',
    'vtc.vn',
    'laodong.vn',
    'sggp.org.vn',
    'nld.com.vn',
    'baochinhphu.vn',
    'hanoimoi.com.vn',
    'giaoducthoidai.vn',
    'qdnd.vn',            # Quân đội nhân dân
    'cand.com.vn',        # Công an nhân dân
    'viettimes.vn',
    'plo.vn',             # Pháp luật TP.HCM
    'anninhthudo.vn',
    'kinhtedothi.vn',
    'kienthuc.net.vn',
    'soha.vn',
    'genk.vn',
    'tinhte.vn',
    'danviet.vn',
    'petrotimes.vn',
    'baotintuc.vn',
    'ngoisao.vn',
    'vietq.vn',
    'baogiaothong.vn',
    'congthuong.vn',
    'antv.gov.vn',
    'vietbao.vn',
    'vietgiaitri.com',
    'vietinfo.vn',
    'doisongphapluat.vn'
]

# Danh sách các domain bị chặn hoặc không đáng tin cậy
BLOCKED_DOMAINS = [
    'facebook.com',
    'twitter.com',
    'instagram.com',
    'tiktok.com',
    'youtube.com',
    'youtu.be',
    'pinterest.com',
    'blogspot.com',
    'wordpress.com',
    'tumblr.com',
    'reddit.com',
    'quora.com',
    'medium.com',
    'telegram.org',
    'messenger.com',
    'whatsapp.com',
    'viber.com',
    'zalo.me',
    'zingmp3.vn',
    'mp3.zing.vn',
    'nhaccuatui.com',
    'soundcloud.com',
    'spotify.com',
    'netflix.com',
    'phimmoi.net',
    'phimbathu.com',
    'bilutv.com',
    'motphim.net',
    'kphim.tv',
    'phimhay.net',
    'pornhub.com',
    'xvideos.com',
    'xnxx.com',
    'xhamster.com',
    'redtube.com',
    'youporn.com'
]

# Bộ chọn CSS để trích xuất nội dung từ các trang tin tức
NEWS_CONTENT_SELECTORS = {
    # Wikipedia
    'vi.wikipedia.org': {'class_': 'mw-parser-output'},
    
    # Các báo lớn
    'vnexpress.net': {'class_': 'article-content'},
    'tuoitre.vn': {'class_': 'detail-content'},
    'thanhnien.vn': {'class_': 'detail__content'},
    'vietnamnet.vn': {'class_': 'content-detail'},
    'dantri.com.vn': {'class_': 'singular-content'},
    'nhandan.vn': {'class_': 'detail-content-body'},
    'tienphong.vn': {'class_': 'article__body'},
    'baodautu.vn': {'class_': 'content-news-detail'},
    'cafef.vn': {'class_': 'detail-content'},
    'zingnews.vn': {'class_': 'the-article-body'},
    'vneconomy.vn': {'class_': 'detail-content'},
    'nguoiduatin.vn': {'class_': 'content-detail'},
    'kenh14.vn': {'class_': 'knc-content'},
    'vov.vn': {'class_': 'article-content'},
    'vtc.vn': {'class_': 'content-detail'},
    'laodong.vn': {'class_': 'article-content'},
    'sggp.org.vn': {'class_': 'article-body'},
    'nld.com.vn': {'class_': 'detail-content'},
    'baochinhphu.vn': {'class_': 'detail-content-body'},
    'hanoimoi.com.vn': {'class_': 'detail-content'},
    'giaoducthoidai.vn': {'class_': 'content-news-detail'},
    
    # Các báo khác
    'qdnd.vn': {'class_': 'post-content'},
    'cand.com.vn': {'class_': 'detail-content'},
    'viettimes.vn': {'class_': 'detail-content'},
    'plo.vn': {'class_': 'article-content'},
    'anninhthudo.vn': {'class_': 'detail-content'},
    'kinhtedothi.vn': {'class_': 'detail-content'},
    'kienthuc.net.vn': {'class_': 'detail-content'},
    'soha.vn': {'class_': 'news-content'},
    'genk.vn': {'class_': 'detail-content'},
    'tinhte.vn': {'class_': 'thread-body'},
    'danviet.vn': {'class_': 'detail-content'},
    'petrotimes.vn': {'class_': 'detail-content'},
    'baotintuc.vn': {'class_': 'article-body'},
    'ngoisao.vn': {'class_': 'article-content'},
    'vietq.vn': {'class_': 'content-detail'},
    'baogiaothong.vn': {'class_': 'detail-content'},
    'congthuong.vn': {'class_': 'detail-content'},
    'antv.gov.vn': {'class_': 'article-content'},
    'vietbao.vn': {'class_': 'detail-content'},
    'vietgiaitri.com': {'class_': 'article-content'},
    'vietinfo.vn': {'class_': 'article-content'},
    'doisongphapluat.vn': {'class_': 'post-content'}
}

def create_service():
    """Tạo dịch vụ Google Custom Search với cache bị vô hiệu hóa"""
    try:
        # Tạo và trả về đối tượng dịch vụ Google Custom Search
        # Chọn ngẫu nhiên một API key từ danh sách để tránh vượt quá giới hạn truy vấn
        return build(
            "customsearch", 
            "v1",
            developerKey=API_KEYS[random.randint(0, len(API_KEYS)-1)],
            cache_discovery=False  # Tắt cache để tránh lỗi với oauth2client
        )
        
    except Exception as e:
        # Ghi log lỗi nếu không thể tạo dịch vụ
        logging.error(f"Error creating service: {e}")
        return None

def create_requests_session():
    """Tạo phiên requests với cơ chế thử lại"""
    session = requests.Session()  # Tạo một phiên mới
    
    # Cấu hình cơ chế thử lại cho các lỗi mạng phổ biến
    retry = Retry(
        total=3,  # Tổng số lần thử lại
        backoff_factor=1,  # Hệ số chờ giữa các lần thử (1s, 2s, 4s...)
        status_forcelist=[500, 502, 503, 504, 404],  # Các mã lỗi HTTP cần thử lại
    )
    
    # Áp dụng cấu hình thử lại cho các kết nối HTTP và HTTPS
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session  # Trả về phiên đã cấu hình

def clean_text(text: str) -> str:
    """Làm sạch và chuẩn hóa văn bản"""
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    
    # Xóa ký tự đặc biệt nhưng giữ lại ký tự tiếng Việt
    text = re.sub(r'[^\w\s.,?!-áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', text, flags=re.IGNORECASE)
    
    return text.strip()  # Xóa khoảng trắng ở đầu và cuối chuỗi

def get_domain(url: str) -> str:
    """Trích xuất tên miền từ URL"""
    # Sử dụng biểu thức chính quy để lấy phần tên miền chính từ URL
    match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url.lower())
    return match.group(1) if match else ''  # Trả về tên miền hoặc chuỗi rỗng nếu không tìm thấy

@lru_cache(maxsize=100)
def cached_getContent(url: str) -> str:
    """Phiên bản có cache của hàm getContent"""
    # Sử dụng decorator lru_cache để lưu trữ kết quả của các URL đã truy cập
    return getContent(url)

def getContent(url: str) -> str:
    """Lấy và xử lý nội dung từ URL với xử lý lỗi nâng cao"""
    # Tạo phiên requests với cơ chế thử lại
    session = create_requests_session()
    
    # Thiết lập các header HTTP để mô phỏng trình duyệt thật
    headers = {
        'User-Agent': random.choice(USER_AGENTS),  # Chọn ngẫu nhiên User-Agent
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
        'Referer': 'https://www.google.com.vn/'  # Giả lập truy cập từ Google
    }

    try:
        # Lấy tên miền từ URL
        domain = get_domain(url)
        
        # Kiểm tra xem domain có trong danh sách bị chặn không
        if any(blocked_domain in domain for blocked_domain in BLOCKED_DOMAINS):
            logging.debug(f"Skipping blocked domain: {domain}")
            return ''
            
        # Kiểm tra xem domain có trong danh sách cho phép không
        if not any(allowed_domain in domain for allowed_domain in ALLOWED_DOMAINS):
            logging.debug(f"Skipping non-allowed domain: {domain}")
            return ''

        # Gửi request HTTP và lấy nội dung trang
        response = session.get(
            url, 
            headers=headers, 
            timeout=20,  # Thời gian chờ tối đa 20 giây
            verify=False,  # Bỏ qua xác minh SSL để tránh lỗi với một số trang
            allow_redirects=True  # Cho phép chuyển hướng
        )
        response.raise_for_status()  # Phát sinh lỗi nếu status code >= 400
        response.encoding = 'utf-8'  # Đặt encoding là utf-8 để xử lý tiếng Việt

        # Phân tích cú pháp HTML bằng BeautifulSoup
        tree = BeautifulSoup(response.text, 'lxml')

        # Xóa các phần tử không mong muốn
        for elem in tree.find_all(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'meta', 'aside', 'noscript']):
            elem.decompose()

        # Thử lấy nội dung sử dụng bộ chọn cụ thể cho từng trang web
        content_selector = NEWS_CONTENT_SELECTORS.get(domain, {})
        main_content = tree.find(['div', 'article'], **content_selector) if content_selector else None

        # Nếu không tìm thấy nội dung bằng bộ chọn cụ thể, thử với các chỉ báo nội dung phổ biến
        if not main_content:
            main_content = tree.find(['div', 'article'], 
                class_=lambda x: x and any(indicator in str(x).lower() 
                    for indicator in ['article-content', 'article-body', 'main-content', 'detail-content', 'content-detail', 'news-content', 'post-content', 'entry-content', 'singular-content', 'content', 'detail']))

        # Phương án dự phòng thứ hai - tìm bất kỳ div nào có nội dung đáng kể
        if not main_content:
            all_divs = tree.find_all('div')
            if all_divs:
                # Sắp xếp các div theo độ dài nội dung và lấy cái dài nhất
                main_content = sorted(all_divs, key=lambda x: len(x.get_text()), reverse=True)[0]
            else:
                main_content = tree  # Nếu không tìm thấy div nào, sử dụng toàn bộ trang

        # Lấy các đoạn văn
        paragraphs = []
        for p in main_content.find_all(['p', 'div']):
            # Bỏ qua nếu nằm trong các phần tử không mong muốn
            if p.find_parent(['div', 'article'], class_=lambda x: x and any(
                term in str(x).lower() for term in ['related', 'advertisement', 'recommend', 'social', 'comment', 'share', 'tag', 'author', 'byline', 'meta', 'footer', 'header', 'sidebar', 'widget'])):
                continue
            
            # Làm sạch và lấy nội dung văn bản
            text = clean_text(p.get_text())
            if text and len(text.split()) > 3:  # Lấy đoạn văn có ít nhất 4 từ
                paragraphs.append(text)

        # Loại bỏ các đoạn trùng lặp nhưng giữ nguyên thứ tự
        seen = set()
        paragraphs = [x for x in paragraphs if not (x in seen or seen.add(x))]

        # Xử lý các câu
        result = []
        for para in paragraphs:
            try:
                # Tách đoạn văn thành các câu riêng biệt
                sents = sent_tokenize(para)
                if sents:
                    # Làm sạch và lọc các câu có độ dài đủ
                    clean_sents = [s.strip() for s in sents if len(s.strip()) > 5]  # Lấy câu có ít nhất 6 ký tự
                    if clean_sents:
                        result.append(' . '.join(clean_sents))
            except Exception as e:
                logging.debug(f"Error tokenizing sentences: {e}")
                # Nếu việc tách câu thất bại, sử dụng đoạn văn nguyên bản
                if len(para) > 10:
                    result.append(para)
                continue

        # Kết hợp tất cả các đoạn văn đã xử lý
        final_text = '\n\n'.join(result)
        
        # Loại bỏ nội dung quá ngắn
        if len(final_text) < 30:
            logging.debug(f"Content too short for {url}: {len(final_text)} chars")
            return ''
            
        return final_text  # Trả về nội dung đã xử lý

    except Exception as e:
        # Ghi log lỗi nếu có vấn đề khi xử lý URL
        logging.error(f"Error processing {url}: {str(e)}")
        return ''
    finally:
        # Đảm bảo phiên luôn được đóng để giải phóng tài nguyên
        session.close()

def ggsearch(para: Tuple[int, Any, str]) -> List[dict]:
    """Thực hiện tìm kiếm Google Custom Search"""
    try:
        i, service, query = para  # Giải nén tham số
        
        # Thiết lập tham số tìm kiếm
        search_params = {
            'q': query,  # Truy vấn tìm kiếm
            'cx': CUSTOM_SEARCH_ENGINE_ID,  # ID của Custom Search Engine
            'gl': 'vn',  # Địa phương hóa tìm kiếm cho Việt Nam
            'googlehost': 'vn',  # Sử dụng máy chủ Google Việt Nam
            'hl': 'vi'  # Ngôn ngữ tiếng Việt
        }
        
        # Nếu không phải trang đầu tiên, thêm tham số phân trang
        if i > 0:
            search_params.update({'num': 10, 'start': i * 10})
            
        # Thực hiện tìm kiếm và trả về kết quả
        res = service.cse().list(**search_params).execute()
        return res.get('items', [])  # Trả về danh sách các mục tìm kiếm hoặc mảng rỗng nếu không có kết quả
    except Exception as e:
        # Ghi log lỗi nếu tìm kiếm thất bại
        logging.error(f"Search error: {e}")
        return []

class GoogleSearch:
    """Lớp Singleton cho các hoạt động tìm kiếm Google"""
    _instance = None  # Biến lưu trữ thể hiện duy nhất của lớp
    
    def __new__(cls):
        # Đảm bảo chỉ có một thể hiện của lớp được tạo
        if cls._instance is None:
            cls._instance = super(GoogleSearch, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Chỉ khởi tạo một lần
        if self._initialized:
            return
            
        # Khởi tạo các thành phần cần thiết
        self.service = create_service()  # Dịch vụ tìm kiếm Google
        self.executor = ThreadPoolExecutor(max_workers=4)  # Bộ thực thi đa luồng
        self.session = create_requests_session()  # Phiên HTTP
        self._initialized = True  # Đánh dấu đã khởi tạo

    def search(self, question: str) -> Tuple[List[str], List[str]]:
        """
        Tìm kiếm Google và trích xuất nội dung từ các trang tin tức tiếng Việt
        Args:
            question: Truy vấn tìm kiếm
        Returns:
            Tuple của (URLs, nội dung tài liệu)
        """
        try:
            # Đảm bảo dịch vụ tìm kiếm đã được khởi tạo
            if not self.service:
                self.service = create_service()
            if not self.service:
                logging.error("Could not create search service")
                return [], []

            logging.info(f"Searching for: {question}")  # Ghi log thông tin tìm kiếm

            # Thử tìm kiếm trực tiếp trước
            pages_content = []  # Lưu trữ kết quả tìm kiếm
            max_retries = 3  # Số lần thử lại tối đa
            
            # Thử tìm kiếm với truy vấn gốc
            for attempt in range(max_retries):
                try:
                    result = ggsearch((0, self.service, question))  # Thực hiện tìm kiếm trang đầu tiên
                    if result:
                        pages_content.extend(result)
                        logging.info(f"Found {len(result)} results with direct search")
                        break
                    else:
                        logging.warning(f"No results found with direct search, attempt {attempt+1}")
                except Exception as e:
                    logging.error(f"Direct search error, attempt {attempt+1}: {e}")
                    continue
            
            # Nếu không tìm thấy kết quả hoặc ít kết quả, thử tìm với site operator
            if len(pages_content) < 5:
                # Tạo site operator với 10 domain đầu tiên
                site_operator = ' OR '.join(f'site:{domain}' for domain in ALLOWED_DOMAINS[:10])
                modified_query = f"({question}) ({site_operator})"  # Kết hợp truy vấn với site operator
                
                # Thử tìm kiếm với truy vấn đã sửa đổi
                for attempt in range(max_retries):
                    try:
                        result = ggsearch((0, self.service, modified_query))
                        if result:
                            pages_content.extend(result)
                            logging.info(f"Found {len(result)} additional results with site operators")
                            break
                    except Exception as e:
                        logging.error(f"Site operator search error, attempt {attempt+1}: {e}")
                        continue

            # Kiểm tra nếu không có kết quả tìm kiếm
            if not pages_content:
                logging.warning("No search results found")
                return [], []

            # Ghi log số lượng kết quả
            logging.info(f"Total search results: {len(pages_content)}")
            
            # Trích xuất URL từ các domain được cho phép
            document_urls = []
            for page in pages_content:
                if 'link' in page and not page.get('fileFormat'):  # Bỏ qua các tệp (PDF, DOC...)
                    url = page['link']
                    domain = get_domain(url)
                    
                    # Kiểm tra xem domain có trong danh sách bị chặn không
                    if any(blocked_domain in domain for blocked_domain in BLOCKED_DOMAINS):
                        logging.debug(f"Skipping blocked URL: {url}")
                        continue
                        
                    # Kiểm tra xem domain có trong danh sách cho phép không
                    if any(allowed_domain in domain for allowed_domain in ALLOWED_DOMAINS):
                        document_urls.append(url)
                    else:
                        logging.debug(f"Skipping non-allowed URL: {url}")

            logging.info(f"Found {len(document_urls)} valid URLs")

            # Nếu không có URL hợp lệ, sử dụng đoạn trích làm phương án dự phòng
            if not document_urls:
                logging.warning("No valid URLs found, using snippets as fallback")
                successful_urls = []
                gg_documents = []
                
                # Lấy đoạn trích từ kết quả tìm kiếm
                for page in pages_content:
                    if 'snippet' in page:
                        snippet = page.get('snippet', '')
                        if len(snippet) > 20:  # Ngưỡng thấp hơn cho đoạn trích
                            gg_documents.append(snippet)
                            successful_urls.append(page.get('link', 'unknown'))
                
                if gg_documents:
                    logging.info(f"Using {len(gg_documents)} snippets as content")
                    return successful_urls, gg_documents
                return [], []

            # Lấy nội dung từ URL với xử lý song song
            gg_documents = []
            successful_urls = []
            
            # Sử dụng ThreadPoolExecutor để tải nhiều trang cùng lúc
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Tạo future cho mỗi URL
                future_to_url = {
                    executor.submit(cached_getContent, url): url 
                    for url in document_urls
                }
                
                # Xử lý kết quả khi hoàn thành
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result(timeout=30)  # Đặt thời gian chờ 30 giây
                        if content:
                            gg_documents.append(content)
                            successful_urls.append(url)
                    except Exception as e:
                        logging.error(f"Error processing {url}: {e}")

            logging.info(f"Successfully extracted content from {len(gg_documents)} URLs")

            # Nếu không thể trích xuất nội dung từ bất kỳ URL nào, sử dụng đoạn trích
            if not gg_documents:
                logging.warning("No documents could be processed, using snippets as fallback")
                for page in pages_content:
                    if 'snippet' in page:
                        snippet = page.get('snippet', '')
                        if len(snippet) > 20:
                            gg_documents.append(snippet)
                            successful_urls.append(page.get('link', 'unknown'))
                
                if gg_documents:
                    logging.info(f"Using {len(gg_documents)} snippets as content")
                    return successful_urls, gg_documents
                return [], []

            # Trả về danh sách URL thành công và nội dung tài liệu
            return successful_urls, gg_documents

        except Exception as e:
            # Ghi log lỗi nếu có vấn đề với toàn bộ hoạt động tìm kiếm
            logging.error(f"Search operation error: {e}")
            return [], []
        
    def __del__(self):
        """Dọn dẹp tài nguyên khi đối tượng bị hủy"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)  # Đóng executor mà không chờ các tác vụ hoàn thành
        if hasattr(self, 'session'):
            self.session.close()  # Đóng phiên HTTP để giải phóng tài nguyên

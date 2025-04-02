# Import các thư viện cần thiết
import re                              # Thư viện để xử lý biểu thức chính quy
import logging                         # Thư viện để ghi log
from typing import Dict, Set, List, Union  # Các kiểu dữ liệu để chú thích kiểu
import os                              # Thư viện để thao tác với hệ thống tệp

# Cấu hình logging cơ bản
# Thiết lập mức độ log là INFO và định dạng log gồm thời gian, cấp độ và nội dung
logging.basicConfig(
    level=logging.INFO,                # Thiết lập mức độ log là INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Định dạng log
)

def cleanhtml(raw_html: str) -> str:
    """
    Hàm loại bỏ các thẻ HTML từ chuỗi đầu vào.
    
    Args:
        raw_html: Chuỗi có thể chứa các thẻ HTML
        
    Returns:
        Chuỗi đã được làm sạch, loại bỏ thẻ HTML và cắt khoảng trắng thừa
    """
    try:
        cleanr = re.compile('<.*?>')   # Tạo biểu thức chính quy để tìm thẻ HTML
        cleantext = re.sub(cleanr, '', raw_html)  # Thay thế thẻ HTML bằng chuỗi rỗng
        return cleantext.strip()       # Trả về chuỗi đã loại bỏ khoảng trắng thừa
    except Exception as e:
        logging.error(f"Error cleaning HTML: {e}")  # Ghi log nếu có lỗi
        return raw_html                # Trả về chuỗi ban đầu nếu có lỗi

def read_file_safe(filepath: str) -> str:
    """
    Hàm đọc tệp an toàn với nhiều bảng mã khác nhau.
    
    Args:
        filepath: Đường dẫn đến tệp cần đọc
        
    Returns:
        Nội dung của tệp dưới dạng chuỗi
        
    Raises:
        Exception: Nếu không thể đọc tệp với bất kỳ bảng mã nào
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']  # Danh sách các bảng mã thử
    for encoding in encodings:         # Thử từng bảng mã
        try:
            with open(filepath, 'r', encoding=encoding) as f:  # Mở tệp với bảng mã hiện tại
                return f.read()        # Trả về nội dung nếu đọc thành công
        except UnicodeDecodeError:     # Bỏ qua lỗi giải mã Unicode
            continue
        except Exception as e:         # Xử lý các lỗi khác
            logging.error(f"Error reading file {filepath} with {encoding}: {e}")
    # Nếu đã thử tất cả bảng mã mà không thành công
    raise Exception(f"Could not read file {filepath} with any encoding")

def loadMap(file_name: str) -> Dict[str, str]:
    """
    Hàm tải bản đồ ánh xạ từ tệp văn bản.
    
    Args:
        file_name: Đường dẫn đến tệp cần tải
        
    Returns:
        Dictionary chứa các cặp khóa-giá trị từ tệp
    """
    try:
        content = read_file_safe(file_name)  # Đọc nội dung tệp
        entities = content.split('\n')       # Tách thành các dòng
        entities = [e.split('\t') for e in entities if e.strip()]  # Tách mỗi dòng thành các phần theo tab
        entities = [e for e in entities if len(e) > 1]  # Chỉ giữ lại các dòng có ít nhất 2 phần
        
        map_: Dict[str, str] = {}            # Khởi tạo dictionary rỗng
        for pair in entities:                # Duyệt qua từng cặp
            clean_key = cleanhtml(pair[0])   # Làm sạch khóa
            clean_value = cleanhtml(pair[1]) # Làm sạch giá trị
            if clean_key and clean_value:    # Nếu cả khóa và giá trị đều không rỗng
                map_[clean_key] = clean_value # Thêm vào dictionary
                
        return map_                          # Trả về dictionary
    except Exception as e:
        logging.error(f"Error loading map from {file_name}: {e}")  # Ghi log nếu có lỗi
        return {}                            # Trả về dictionary rỗng nếu có lỗi

# Tải tài nguyên
try:
    # Xây dựng đường dẫn đến tệp tài nguyên
    resource_path = os.path.join('resources', 'lower_vi_syns.txt')
    # Tải bản đồ từ đồng nghĩa tiếng Việt
    el_vi = loadMap(resource_path)
    
    # Khởi tạo dictionary chứa các biến thể của mỗi thực thể
    el_map_variants: Dict[str, Set[str]] = {}
    # Khởi tạo dictionary ngược - từ giá trị đến danh sách các khóa
    r_vi: Dict[str, List[str]] = {}

    # Duyệt qua từng cặp khóa-giá trị trong bản đồ từ đồng nghĩa
    for e in el_vi:
        # Khởi tạo tập hợp rỗng cho khóa
        el_map_variants[e] = set()
        # Khởi tạo tập hợp rỗng cho giá trị
        el_map_variants[el_vi[e]] = set()
        
        # Xây dựng bản đồ ngược: từ giá trị đến danh sách các khóa
        if el_vi[e] not in r_vi:
            r_vi[el_vi[e]] = [e]      # Nếu giá trị chưa có trong r_vi, tạo danh sách mới
        else:
            r_vi[el_vi[e]].append(e)  # Nếu đã có, thêm khóa vào danh sách hiện tại
            
    # Duyệt lại qua từng cặp để xây dựng mạng lưới các biến thể
    for e in el_vi:
        entity = e                     # Thực thể gốc (khóa)
        o_entity = el_vi[e]            # Thực thể đích (giá trị)

        # Thêm mỗi thực thể vào tập biến thể của thực thể kia
        el_map_variants[entity].add(o_entity)
        el_map_variants[o_entity].add(entity)

        # Nếu thực thể gốc có trong bản đồ ngược
        if entity in r_vi:
            # Thêm tất cả các biến thể của entity vào cả entity và o_entity
            el_map_variants[entity] = el_map_variants[entity].union(set(r_vi[entity]))
            el_map_variants[o_entity] = el_map_variants[o_entity].union(set(r_vi[entity]))

        # Nếu thực thể đích có trong bản đồ ngược
        if o_entity in r_vi:
            # Thêm tất cả các biến thể của o_entity vào cả entity và o_entity
            el_map_variants[entity] = el_map_variants[entity].union(set(r_vi[o_entity]))
            el_map_variants[o_entity] = el_map_variants[o_entity].union(set(r_vi[o_entity]))

    # Xóa bản đồ ngược để giải phóng bộ nhớ
    del r_vi
    
except Exception as e:
    # Xử lý lỗi khi khởi tạo liên kết thực thể
    logging.error(f"Error initializing entity linking: {e}")
    # Khởi tạo các dictionary rỗng nếu có lỗi
    el_vi = {}
    el_map_variants = {}

def getVariants(entity: Union[str, object]) -> List[str]:
    """
    Hàm lấy các biến thể của một thực thể.
    
    Args:
        entity: Thực thể cần lấy các biến thể, có thể là chuỗi hoặc đối tượng
        
    Returns:
        Danh sách các biến thể của thực thể
    """
    try:
        # Chuyển đối tượng thành chuỗi nếu chưa phải chuỗi
        if not isinstance(entity, str):
            entity = str(entity)

        # Chuẩn hóa thực thể: thay thế gạch dưới bằng khoảng trắng, chuyển thành chữ thường và loại bỏ khoảng trắng thừa
        entity = entity.replace('_', ' ').lower().strip()
        variants = [entity]            # Khởi tạo danh sách biến thể với chính thực thể
        # Nếu thực thể có trong bản đồ biến thể
        if entity in el_map_variants:
            variants = list(el_map_variants[entity])  # Lấy tất cả biến thể
            
        return list(set(variants))     # Trả về danh sách các biến thể đã loại bỏ trùng lặp
    except Exception as e:
        logging.error(f"Error getting variants for {entity}: {e}")  # Ghi log nếu có lỗi
        return [str(entity)]           # Trả về danh sách chỉ chứa thực thể ban đầu nếu có lỗi

def extractEntVariants(entity: Union[str, object]) -> List[str]:
    """
    Hàm trích xuất tất cả các biến thể của một thực thể, bao gồm cả thực thể gốc.
    
    Args:
        entity: Thực thể cần trích xuất biến thể
        
    Returns:
        Danh sách các biến thể, bao gồm cả thực thể gốc
    """
    try:
        variants = getVariants(entity)  # Lấy các biến thể của thực thể
        variants.append(str(entity))    # Thêm thực thể gốc vào danh sách
        return list(set(variants))      # Trả về danh sách đã loại bỏ trùng lặp
    except Exception as e:
        logging.error(f"Error extracting variants for {entity}: {e}")  # Ghi log nếu có lỗi
        return [str(entity)]            # Trả về danh sách chỉ chứa thực thể ban đầu nếu có lỗi

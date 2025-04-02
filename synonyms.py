from underthesea import pos_tag  # Import hàm gán nhãn từ loại từ thư viện underthesea
import string  # Import module xử lý chuỗi
import functools  # Import module hỗ trợ các hàm bậc cao
import json  # Import module xử lý JSON
import itertools  # Import module hỗ trợ tạo tổ hợp
import os  # Import module tương tác hệ thống

def read_file_with_encoding(file_path):
    """Đọc file với các encoding khác nhau để tìm encoding phù hợp"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']  # Danh sách các encoding phổ biến
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:  # Thử mở file với encoding hiện tại
                return f.read()  # Trả về nội dung file nếu đọc thành công
        except UnicodeDecodeError:
            continue  # Thử encoding tiếp theo nếu có lỗi
    raise Exception(f"Could not read file {file_path} with any encoding")  # Báo lỗi nếu không đọc được với bất kỳ encoding nào

# Tải danh sách stopwords (từ dừng)
try:
    stopwords = read_file_with_encoding('resources/stopwords_small.txt').split('\n')  # Đọc file stopwords và tách theo dòng
    stopwords = set([w.replace(' ','_') for w in stopwords])  # Thay thế khoảng trắng bằng dấu gạch dưới và chuyển thành set
except Exception as e:
    print(f"Warning: Error loading stopwords: {e}")  # In cảnh báo nếu có lỗi
    stopwords = set()  # Khởi tạo set rỗng nếu không đọc được file

# Định nghĩa tập hợp các ký tự dấu câu
punct_chars = [c for c in string.punctuation] + ['"','"',"...","–","…","..","•",'"','"']  # Danh sách dấu câu mở rộng
punct_set = set(punct_chars)  # Chuyển thành set để tìm kiếm nhanh hơn

# Tải dữ liệu bigram (cặp từ)
try:
    data = read_file_with_encoding('resources/bigram.txt').split('\n')  # Đọc file bigram và tách theo dòng
    data = [x for x in data if x.strip()]  # Loại bỏ dòng trống
    
    markov_score = {}  # Dictionary lưu trữ điểm số Markov
    for line in data:
        try:
            word, score = line.split('\t')  # Tách dòng thành từ và điểm số
            markov_score[word] = int(score)  # Lưu điểm số vào dictionary
        except:
            continue  # Bỏ qua dòng không đúng định dạng
    del data  # Giải phóng bộ nhớ
except Exception as e:
    print(f"Warning: Error loading bigram data: {e}")  # In cảnh báo nếu có lỗi
    markov_score = {}  # Khởi tạo dictionary rỗng nếu không đọc được file

def makovCal(a, b):
    """Tính điểm Markov cho cặp từ (bigram)"""
    try:
        termBigram = a + "_" + b  # Tạo chuỗi bigram
        
        freBigram = markov_score.get(termBigram, 1)  # Lấy tần suất của bigram, mặc định là 1
        freUnigram = markov_score.get(a, 1)  # Lấy tần suất của từ đầu tiên, mặc định là 1
        
        if freUnigram < 5:
            freUnigram = 2000  # Đặt giá trị mặc định nếu từ quá hiếm
        else:
            freUnigram += 2000  # Thêm giá trị cơ sở để tránh chia cho số quá nhỏ
            
        return float(freBigram) / freUnigram  # Tính xác suất có điều kiện
    except:
        return 0  # Trả về 0 nếu có lỗi

# Định nghĩa ánh xạ từ loại
map_pos = {
    'M':'noun', 'Y':'noun','Nb':'noun','Nc':'noun','Ni':'noun',  # Các loại danh từ
    'Np':'noun','N':'noun','X':'adj','Nu':'noun','Ny':'noun',
    'V':'verb', 'Vb':'verb','Vy':'verb','A': 'adj','Ab': 'adj','R':'adj'  # Động từ, tính từ
}

# Tải từ điển đồng nghĩa
try:
    with open('resources/synonym.json', 'r', encoding='utf-8') as f:  # Mở file JSON
        map_synonym = json.load(f)  # Đọc dữ liệu JSON
except Exception as e:
    print(f"Warning: Error loading synonym dictionary: {e}")  # In cảnh báo nếu có lỗi
    map_synonym = {}  # Khởi tạo dictionary rỗng nếu không đọc được file

def generateCombinations(tokens, thresh_hold):
    """Tạo các tổ hợp từ với từ đồng nghĩa"""
    try:
        combinations = []  # Danh sách lưu các tổ hợp từ
        for i in range(0, len(tokens)):  # Duyệt qua từng token
            word = tokens[i][0].lower()  # Lấy từ và chuyển thành chữ thường
            
            if word in stopwords:
                combinations.append([word])  # Nếu là stopword, giữ nguyên
                continue
            
            pos = tokens[i][1]  # Lấy nhãn từ loại
            if pos in map_pos:  # Nếu từ loại được hỗ trợ
                pos = map_pos[pos]  # Chuyển đổi nhãn từ loại
                if word in map_synonym.get(pos, {}):  # Nếu từ có trong từ điển đồng nghĩa
                    synonyms = map_synonym[pos][word]  # Lấy danh sách từ đồng nghĩa
                    possible_synonym = []  # Danh sách từ đồng nghĩa phù hợp
                    
                    for syn in synonyms:  # Duyệt qua từng từ đồng nghĩa
                        pre_word = tokens[i-1][0].lower() if i > 0 else 'NONE'  # Lấy từ trước đó
                        next_word = tokens[i+1][0].lower() if i < len(tokens)-1 else 'NONE'  # Lấy từ kế tiếp

                        # Kiểm tra điểm Markov với từ trước và từ sau
                        if makovCal(pre_word, syn) > thresh_hold or makovCal(syn, next_word) > thresh_hold:
                            possible_synonym.append(syn)  # Thêm vào danh sách nếu vượt ngưỡng
                    
                    combinations.append([word] + possible_synonym)  # Thêm từ gốc và các từ đồng nghĩa
                else:
                    combinations.append([word])  # Nếu không có từ đồng nghĩa, giữ nguyên
            else:
                combinations.append([word])  # Nếu từ loại không được hỗ trợ, giữ nguyên

        return combinations  # Trả về danh sách các tổ hợp
    except Exception as e:
        print(f"Error in generateCombinations: {e}")  # In lỗi nếu có
        return [[word] for word, _ in tokens]  # Trả về danh sách mặc định nếu có lỗi

def generateVariants(untokenize_text):
    """Tạo các biến thể của văn bản sử dụng từ đồng nghĩa"""
    try:
        # Tokenize và chuẩn hóa văn bản
        words = pos_tag(untokenize_text)  # Gán nhãn từ loại cho văn bản
        words = [(w[0].replace(' ','_'), w[1]) for w in words]  # Thay thế khoảng trắng bằng dấu gạch dưới
        
        # Tạo các tổ hợp với ngưỡng ban đầu
        combinations = generateCombinations(words, 0.001)  # Tạo tổ hợp với ngưỡng 0.001
        num_variants = functools.reduce(lambda x, y: x*y, [len(c) for c in combinations])  # Tính số lượng biến thể
        
        # Điều chỉnh ngưỡng nếu có quá nhiều biến thể
        base_line = 0.001  # Ngưỡng ban đầu
        while num_variants > 10000:  # Nếu có quá nhiều biến thể
            base_line *= 2  # Tăng ngưỡng lên gấp đôi
            combinations = generateCombinations(words, base_line)  # Tạo lại tổ hợp với ngưỡng mới
            num_variants = functools.reduce(lambda x, y: x*y, [len(c) for c in combinations])  # Tính lại số lượng biến thể
        
        # Tạo các tổ hợp cuối cùng
        combinations = list(itertools.product(*combinations))  # Tạo tất cả các tổ hợp có thể
        return [' '.join(e) for e in combinations]  # Nối các từ thành chuỗi và trả về danh sách
    except Exception as e:
        print(f"Error in generateVariants: {e}")  # In lỗi nếu có
        return [untokenize_text]  # Trả về văn bản gốc nếu có lỗi

# Hàm hỗ trợ kiểm tra encoding của file
def check_resource_files():
    """Kiểm tra xem các file tài nguyên có tồn tại và encoding của chúng"""
    files = [
        'resources/stopwords_small.txt',  # File stopwords
        'resources/bigram.txt',  # File bigram
        'resources/synonym.json'  # File từ điển đồng nghĩa
    ]
    
    for file_path in files:
        if not os.path.exists(file_path):  # Kiểm tra file có tồn tại không
            print(f"Warning: {file_path} does not exist")  # In cảnh báo nếu không tồn tại
            continue
            
        try:
            with open(file_path, 'rb') as f:  # Mở file ở chế độ nhị phân
                content = f.read()  # Đọc nội dung file
                import chardet  # Import thư viện phát hiện encoding
                result = chardet.detect(content)  # Phát hiện encoding
                print(f"File {file_path}: {result}")  # In kết quả
        except Exception as e:
            print(f"Error checking {file_path}: {e}")  # In lỗi nếu có

# Tùy chọn: Kiểm tra các file tài nguyên khi module được import
if __name__ == "__main__":  # Nếu file được chạy trực tiếp
    check_resource_files()  # Kiểm tra các file tài nguyên

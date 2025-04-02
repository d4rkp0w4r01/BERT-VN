from __future__ import absolute_import, division, print_function  # Đảm bảo tương thích với Python 2 và 3

# Import các thư viện cần thiết
import torch  # Thư viện PyTorch cho deep learning
import logging  # Thư viện ghi log
import sys  # Thư viện hệ thống
import traceback  # Thư viện theo dõi lỗi
import numpy as np  # Thư viện xử lý số học
import random  # Thư viện tạo số ngẫu nhiên
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)  # Các lớp xử lý dữ liệu của PyTorch
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering  # Mô hình BERT cho trả lời câu hỏi
from pytorch_pretrained_bert.tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)  # Các lớp tokenizer của BERT
from utils import *  # Import tất cả các hàm từ module utils
from multiprocessing import Process, Pool  # Thư viện đa luồng

# Cấu hình logging nếu chưa được cấu hình
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,  # Cấu hình mức độ log là INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # Định dạng log với thời gian, mức độ và nội dung
    )

logger = logging.getLogger(__name__)  # Tạo logger cho module hiện tại

class Args:
    """
    Lớp lưu trữ các tham số cấu hình cho mô hình BERT
    """
    bert_model = './resources'  # Đường dẫn đến thư mục chứa mô hình BERT đã được huấn luyện
    max_seq_length = 160  # Độ dài tối đa của chuỗi đầu vào (số token)
    doc_stride = 160  # Số token chồng lấp giữa các đoạn văn bản dài
    predict_batch_size = 20  # Kích thước batch khi dự đoán
    n_best_size = 20  # Số lượng dự đoán tốt nhất để xem xét
    max_answer_length = 30  # Độ dài tối đa của câu trả lời (số token)
    verbose_logging = False  # Có ghi log chi tiết hay không
    no_cuda = True  # Không sử dụng GPU (True = chỉ dùng CPU)
    seed = 42  # Giá trị seed cho tính tái lập
    do_lower_case = True  # Chuyển văn bản thành chữ thường
    version_2_with_negative = True  # Hỗ trợ câu hỏi không có câu trả lời (SQuAD v2.0)
    null_score_diff_threshold = 0.0  # Ngưỡng điểm cho câu trả lời "không có câu trả lời"
    max_query_length = 64  # Độ dài tối đa của câu hỏi (số token)
    THRESH_HOLD = 0.95  # Ngưỡng điểm cho việc chấp nhận câu trả lời
    
args = Args()  # Tạo đối tượng Args để sử dụng

# Thiết lập các seed ngẫu nhiên để đảm bảo tính tái lập
random.seed(args.seed)  # Thiết lập seed cho module random
np.random.seed(args.seed)  # Thiết lập seed cho numpy
torch.manual_seed(args.seed)  # Thiết lập seed cho PyTorch

class Reader():
    """
    Lớp Reader sử dụng mô hình BERT để trích xuất câu trả lời từ đoạn văn bản
    """
    def __init__(self):
        """
        Khởi tạo mô hình BERT cho việc trả lời câu hỏi
        """
        self.log = {}  # Dictionary lưu trữ thông tin log
        
        # Xác định thiết bị tính toán (CPU hoặc GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        logger.info(f"Using device: {self.device}")  # Ghi log thiết bị đang sử dụng
        
        try:
            # Tải tokenizer BERT
            logger.info("Loading tokenizer...")  # Ghi log đang tải tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
            
            # Tải mô hình BERT
            logger.info("Loading model...")  # Ghi log đang tải mô hình
            self.model = BertForQuestionAnswering.from_pretrained(args.bert_model)  # Tải mô hình BERT cho trả lời câu hỏi
            self.model.to(self.device)  # Chuyển mô hình sang thiết bị tính toán (CPU/GPU)
            self.model.eval()  # Đặt mô hình ở chế độ đánh giá (không cập nhật trọng số)
            
            self.args = args  # Lưu tham số cấu hình
            logger.info("Reader initialization complete")  # Ghi log hoàn thành khởi tạo
        except Exception as e:
            # Xử lý ngoại lệ khi khởi tạo
            logger.error(f"Error initializing Reader: {e}")  # Ghi log lỗi
            logger.error(traceback.format_exc())  # Ghi log stack trace
            raise  # Ném lại ngoại lệ
    
    def getPredictions(self, question, paragraphs):
        """
        Lấy dự đoán câu trả lời cho một câu hỏi từ các đoạn văn bản
        
        Args:
            question (str): Câu hỏi cần trả lời
            paragraphs (list): Danh sách các đoạn văn bản để tìm câu trả lời
            
        Returns:
            list: Danh sách các cặp [câu trả lời, điểm số] cho mỗi đoạn văn
        """
        try:
            # Kiểm tra tính hợp lệ của dữ liệu đầu vào
            if not question or not question.strip():  # Nếu câu hỏi trống
                logger.warning("Empty question provided")  # Ghi log cảnh báo
                return []  # Trả về danh sách rỗng
                
            if not paragraphs:  # Nếu không có đoạn văn bản nào
                logger.warning("No paragraphs provided to reader")  # Ghi log cảnh báo
                return []  # Trả về danh sách rỗng
            
            # Ghi log kích thước đầu vào
            logger.info(f"Processing question with {len(paragraphs)} paragraphs")
            
            # Tiền xử lý dữ liệu đầu vào
            question = question.replace('_', ' ')  # Thay thế dấu gạch dưới bằng khoảng trắng trong câu hỏi
            # Lọc và xử lý các đoạn văn bản
            paragraphs = [p.replace('_', ' ') for p in paragraphs if p and isinstance(p, str)]
            
            # Kiểm tra xem còn đoạn văn bản nào sau khi lọc không
            if not paragraphs:
                logger.warning("No valid paragraphs after preprocessing")  # Ghi log cảnh báo
                return []  # Trả về danh sách rỗng
            
            # Lấy dự đoán sử dụng mô hình BERT
            logger.info("Running BERT prediction...")  # Ghi log bắt đầu dự đoán
            predictions = predict(question, paragraphs, self.model, self.tokenizer, self.device, self.args)
            
            if not predictions:  # Nếu không có dự đoán nào
                logger.warning("No predictions returned from model")  # Ghi log cảnh báo
                return []  # Trả về danh sách rỗng
            
            # Định dạng lại các dự đoán
            formatted_predictions = []
            for p in predictions:
                if not p:  # Bỏ qua các dự đoán rỗng
                    continue
                    
                # Trích xuất giá trị và chuyển đổi thành chuỗi
                values = list(p.values())  # Lấy các giá trị từ dictionary dự đoán
                values = [str(i) for i in values]  # Chuyển đổi tất cả thành chuỗi
                
                # Chỉ lấy hai phần tử đầu tiên (câu trả lời và điểm số)
                if len(values) >= 2:
                    formatted_predictions.append(values[:2])
            
            logger.info(f"Generated {len(formatted_predictions)} predictions")  # Ghi log số lượng dự đoán
            
            # Dọn dẹp để giúp quản lý bộ nhớ
            del question, paragraphs
            
            return formatted_predictions  # Trả về danh sách dự đoán đã định dạng
            
        except Exception as e:
            # Xử lý ngoại lệ khi dự đoán
            logger.error(f"Error in getPredictions: {str(e)}")  # Ghi log lỗi
            logger.error(traceback.format_exc())  # Ghi log stack trace
            return []  # Trả về danh sách rỗng
    
    def __del__(self):
        """
        Dọn dẹp tài nguyên khi đối tượng bị xóa
        """
        try:
            # Xóa cache CUDA nếu đang sử dụng GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
        except:
            pass  # Bỏ qua lỗi nếu có

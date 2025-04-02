from random import randint  # Import hàm tạo số ngẫu nhiên
from scipy import sparse  # Import thư viện xử lý ma trận thưa
from multiprocessing import Pool  # Import lớp Pool để xử lý đa luồng
from collections import Counter  # Import Counter để đếm tần suất
import gc, string  # Import thư viện quản lý bộ nhớ và xử lý chuỗi
import requests, time  # Import thư viện HTTP và xử lý thời gian
import numpy  # Import thư viện tính toán số học
import re  # Import thư viện biểu thức chính quy
from math import log  # Import hàm logarit
import logging  # Import thư viện ghi log
import timeout_decorator  # Import decorator xử lý timeout
from bs4 import BeautifulSoup  # Import thư viện phân tích HTML
import pickle  # Import thư viện lưu trữ đối tượng
import traceback  # Import thư viện theo dõi lỗi
import os  # Import thư viện tương tác hệ thống

# Cấu hình logging nếu chưa được cấu hình
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,  # Thiết lập mức độ log là INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # Định dạng log với thời gian, mức độ và nội dung
    )

logger = logging.getLogger(__name__)  # Tạo logger cho module hiện tại

# Thử import thư viện underthesea (NLP cho tiếng Việt) với xử lý fallback
try:
    from underthesea import pos_tag, ner, word_tokenize  # Import các hàm xử lý ngôn ngữ tiếng Việt
except ImportError:
    # Nếu không có underthesea, tạo các hàm đơn giản thay thế
    logger.warning("underthesea not found. Using simplified tokenization.")
    def pos_tag(text):
        # Trả về danh sách từ với nhãn 'N' (danh từ) mặc định
        return [(word, 'N') for word in text.split()]
        
    def ner(text):
        # Trả về danh sách từ với nhãn 'O' (không phải thực thể) mặc định
        return [(word, 'O', 'O', 'O') for word in text.split()]
        
    def word_tokenize(text):
        # Tách từ đơn giản dựa trên khoảng trắng
        return text.split()

# Thử import module tạo biến thể từ đồng nghĩa
try:
    from synonyms import generateVariants  # Import hàm tạo biến thể từ đồng nghĩa
except ImportError:
    # Nếu không có module synonyms, tạo hàm đơn giản thay thế
    logger.warning("synonyms module not found. Using identity function.")
    def generateVariants(text):
        # Trả về danh sách chỉ chứa văn bản gốc
        return [text]

# Thử import module liên kết thực thể
try:
    from entity_linking import extractEntVariants  # Import hàm trích xuất biến thể thực thể
except ImportError:
    # Nếu không có module entity_linking, tạo hàm đơn giản thay thế
    logger.warning("entity_linking module not found. Using identity function.")
    def extractEntVariants(entity):
        # Trả về danh sách chỉ chứa thực thể gốc
        return [entity]

# Thử tải mô hình word2vec nếu có sẵn
w2v_available = False  # Biến đánh dấu tính khả dụng của word2vec
try:
    from gensim.models import KeyedVectors  # Import lớp KeyedVectors từ gensim
    if os.path.exists('resources/word2vec-200'):  # Kiểm tra xem file mô hình có tồn tại không
        w2v = KeyedVectors.load('resources/word2vec-200')  # Tải mô hình word2vec
        w2v_available = True  # Đánh dấu word2vec khả dụng
        logger.info("Word2Vec model loaded successfully")  # Ghi log thành công
    else:
        # Nếu không tìm thấy file mô hình
        logger.warning("Word2Vec model file not found. Word embedding features disabled.")
except ImportError:
    # Nếu không có gensim
    logger.warning("gensim not found. Word embedding features disabled.")

# Tải danh sách stopwords với xử lý fallback
try:
    # Đọc file stopwords từ đường dẫn cố định
    stopwords = open('resources/stopwords_small.txt', encoding='utf-8').read().split('\n')
    stopwords = set([w.replace(' ','_') for w in stopwords])  # Chuyển đổi khoảng trắng thành dấu gạch dưới
    logger.info(f"Loaded {len(stopwords)} stopwords")  # Ghi log số lượng stopwords
except FileNotFoundError:
    # Nếu không tìm thấy file stopwords
    logger.warning("Stopwords file not found. Using empty stopwords list.")
    stopwords = set()  # Sử dụng tập rỗng

# Tạo tập hợp các ký tự dấu câu
punct_set = set([c for c in string.punctuation]) | set(['"','"',"...","–","…","..","•",'"','"'])

def cos_sim(a, b):
    """Tính độ tương đồng cosine giữa hai vector"""
    try:
        # Công thức tính độ tương đồng cosine: dot(a,b)/(|a|*|b|)
        return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))
    except:
        return 0  # Trả về 0 nếu có lỗi (ví dụ: vector rỗng)

def document_vector(doc):
    """Tính vector văn bản từ các vector từ"""
    if not w2v_available:
        return numpy.zeros(200)  # Trả về vector không nếu word2vec không khả dụng
        
    try:
        # Lọc các từ có trong từ điển word2vec
        valid_words = [i for i in doc if i in w2v.wv.vocab]
        if not valid_words:
            return numpy.zeros(200)  # Trả về vector không nếu không có từ hợp lệ
            
        # Lấy vector cho mỗi từ hợp lệ
        vec = [w2v.wv[i] for i in valid_words]
        return numpy.sum(vec, axis=0)  # Tính tổng các vector từ
    except Exception as e:
        logger.error(f"Error in document_vector: {e}")  # Ghi log lỗi
        return numpy.zeros(200)  # Trả về vector không nếu có lỗi

def embedding_similarity(s1, s2):
    """Tính độ tương đồng nhúng từ giữa hai văn bản"""
    if not w2v_available:
        return 0  # Trả về 0 nếu word2vec không khả dụng
        
    try:
        # Tách và chuẩn hóa các từ trong hai văn bản
        s1 = list(set(s1.lower().split()))  # Chuyển thành chữ thường, tách từ và loại bỏ trùng lặp
        s2 = list(set(s2.lower().split()))
        
        # Lọc các từ có trong từ điển word2vec
        s1 = [word for word in s1 if word in w2v.wv.vocab]
        s2 = [word for word in s2 if word in w2v.wv.vocab]
        
        if len(s1) == 0 or len(s2) == 0:
            return 0  # Trả về 0 nếu một trong hai văn bản không có từ hợp lệ
        
        # Tính độ tương đồng cosine giữa hai vector văn bản
        return cos_sim(document_vector(s1), document_vector(s2))
    except Exception as e:
        logger.error(f"Error in embedding_similarity: {e}")  # Ghi log lỗi
        return 0  # Trả về 0 nếu có lỗi

def generateNgram(paper, ngram=2, deli='_', rmSet=None):
    """Tạo n-gram từ văn bản"""
    if rmSet is None:
        rmSet = {}  # Tập hợp các từ cần loại bỏ
        
    try:
        words = paper.split()  # Tách văn bản thành các từ
        if len(words) < ngram:
            return []  # Trả về danh sách rỗng nếu văn bản có ít từ hơn n
        
        ngrams = []
        # Duyệt qua văn bản để tạo các n-gram
        for i in range(0, len(words) - ngram + 1):
            block = words[i:i + ngram]  # Lấy n từ liên tiếp
            if not any(w in rmSet for w in block):  # Kiểm tra không có từ nào trong tập loại bỏ
                ngrams.append(deli.join(block))  # Nối các từ bằng dấu phân cách
                
        return ngrams
    except Exception as e:
        logger.error(f"Error in generateNgram: {e}")  # Ghi log lỗi
        return []  # Trả về danh sách rỗng nếu có lỗi

def generatePassages(document, n=3):
    """Chia văn bản thành các đoạn có n câu"""
    try:
        if not document:
            return []  # Trả về danh sách rỗng nếu văn bản trống
            
        passages = []
        paragraphs = document.split('\n\n')  # Tách văn bản thành các đoạn văn
        
        for para in paragraphs:
            if not para.strip():
                continue  # Bỏ qua đoạn văn trống
                
            sentences = para.rsplit(' . ')  # Tách đoạn văn thành các câu
            
            if len(sentences) <= n:
                passages.append(' '.join(sentences))  # Nếu đoạn văn có ít hơn hoặc bằng n câu, giữ nguyên
            else:
                # Nếu đoạn văn có nhiều hơn n câu, tạo các đoạn chồng lấp
                for i in range(0, len(sentences) - n + 1):
                    # Tạo đoạn từ n câu liên tiếp, bỏ qua câu có dấu ?
                    passage = ' '.join([sentences[i + j] for j in range(0, n) if '?' not in sentences[i + j]])
                    if passage.strip():
                        passages.append(passage)
        
        return passages
    except Exception as e:
        logger.error(f"Error in generatePassages: {e}")  # Ghi log lỗi
        return [document] if document else []  # Trả về văn bản gốc nếu có lỗi

def passage_score(q_ngrams, passage):
    """Tính điểm liên quan của đoạn văn với câu hỏi"""
    try:
        if not passage or not q_ngrams:
            return 0  # Trả về 0 nếu đoạn văn hoặc n-gram câu hỏi trống
            
        passage = passage.lower()  # Chuyển đoạn văn thành chữ thường

        # Tính độ trùng lặp unigram (từ đơn)
        p_unigram = set(generateNgram(passage, 1, '_', punct_set | stopwords))  # Tạo unigram từ đoạn văn
        uni_score = len(p_unigram & q_ngrams.get('unigram', set()))  # Đếm số unigram trùng lặp

        # Tính độ trùng lặp n-gram
        p_bigram = set(generateNgram(passage, 2, '_', punct_set | stopwords))  # Tạo bigram từ đoạn văn
        p_trigram = set(generateNgram(passage, 3, '_', punct_set | stopwords))  # Tạo trigram từ đoạn văn
        p_fourgram = set(generateNgram(passage, 4, '_', punct_set))  # Tạo 4-gram từ đoạn văn

        # Đếm số n-gram trùng lặp
        bi_score = len(p_bigram & q_ngrams.get('bigram', set()))  # Đếm số bigram trùng lặp
        tri_score = len(p_trigram & q_ngrams.get('trigram', set()))  # Đếm số trigram trùng lặp
        four_score = len(p_fourgram & q_ngrams.get('fourgram', set()))  # Đếm số 4-gram trùng lặp

        # Tính độ tương đồng nhúng từ nếu có word2vec
        emd_sim = 0
        if w2v_available and p_unigram and 'unigram' in q_ngrams and q_ngrams['unigram']:
            emd_sim = embedding_similarity(' '.join(p_unigram), ' '.join(q_ngrams['unigram']))

        # Tính điểm cuối cùng với trọng số
        return uni_score + bi_score*2 + tri_score*3 + four_score*4 + emd_sim*3
    except Exception as e:
        logger.error(f"Error scoring passage: {e}")  # Ghi log lỗi
        return 0  # Trả về 0 nếu có lỗi

def passage_score_wrap(args):
    """Wrapper cho hàm passage_score để sử dụng với multiprocessing"""
    try:
        return passage_score(args[0], args[1])  # Gọi hàm passage_score với tham số từ tuple
    except Exception as e:
        logger.error(f"Error in passage_score_wrap: {e}")  # Ghi log lỗi
        return 0  # Trả về 0 nếu có lỗi

def chunks(l, n):
    """Chia danh sách thành các phần có kích thước n"""
    for i in range(0, len(l), n):
        yield l[i:i + n]  # Trả về từng phần của danh sách

def get_entities(seq):
    """Trích xuất thực thể có tên từ chuỗi các thẻ"""
    try:
        i = 0
        chunks = []
        seq = seq + ['O']  # Thêm sentinel
        types = [tag.split('-')[-1] for tag in seq]  # Lấy loại thực thể từ thẻ
        while i < len(seq):
            if seq[i].startswith('B'):  # Bắt đầu thực thể mới
                for j in range(i+1, len(seq)):
                    if seq[j].startswith('I') and types[j] == types[i]:  # Tiếp tục thực thể hiện tại
                        continue
                    break
                chunks.append((types[i], i, j))  # Thêm thực thể vào danh sách
                i = j
            else:
                i += 1
        return chunks
    except Exception as e:
        logger.error(f"Error in get_entities: {e}")  # Ghi log lỗi
        return []  # Trả về danh sách rỗng nếu có lỗi

def get_ner(text):
    """Trích xuất thực thể có tên từ văn bản"""
    try:
        res = ner(text)  # Gọi hàm NER từ underthesea
        words = [r[0] for r in res]  # Lấy danh sách từ
        tags = [t[3] for t in res]  # Lấy danh sách thẻ NER
        
        chunks = get_entities(tags)  # Lấy các thực thể từ thẻ
        res = []
        for chunk_type, chunk_start, chunk_end in chunks:
            res.append(' '.join(words[chunk_start: chunk_end]))  # Tạo chuỗi thực thể
        return res
    except Exception as e:
        logger.error(f"Error in get_ner: {e}")  # Ghi log lỗi
        return []  # Trả về danh sách rỗng nếu có lỗi

def keyword_extraction(question):
    """Trích xuất từ khóa từ câu hỏi"""
    try:
        if not question:
            return []  # Trả về danh sách rỗng nếu câu hỏi trống
            
        keywords = []
        question = question.replace('_', ' ')  # Thay thế dấu gạch dưới bằng khoảng trắng
        
        # Thêm chỉ thị so sánh nhất
        if 'nhất' in question.lower():
            keywords.append('nhất')

        # Trích xuất từ với các thẻ POS cụ thể
        try:
            words = pos_tag(question)  # Gán nhãn từ loại cho câu hỏi
            for i in range(0, len(words)):
                words[i] = (words[i][0].replace(' ', '_'), words[i][1])  # Thay thế khoảng trắng bằng dấu gạch dưới
                
            for token in words:
                word = token[0]
                pos = token[1]
                if (pos in ['A', 'Ab']):  # Nếu là tính từ
                    keywords += word.lower().split('_')  # Thêm vào danh sách từ khóa
        except Exception as e:
            logger.warning(f"Error in POS tagging: {e}")  # Ghi log cảnh báo
        
        keywords = list(set(keywords))  # Loại bỏ trùng lặp
        keywords = [[w] for w in keywords]  # Chuyển đổi thành danh sách của danh sách
        
        # Trích xuất thực thể có tên
        try:
            ners = get_ner(question)  # Trích xuất thực thể có tên từ câu hỏi
            ners = [n.lower() for n in ners]  # Chuyển thành chữ thường
            
            for ne in ners:
                variants = extractEntVariants(ne)  # Tạo các biến thể của thực thể
                keywords.append(variants)  # Thêm vào danh sách từ khóa
        except Exception as e:
            logger.warning(f"Error in NER extraction: {e}")  # Ghi log cảnh báo
        
        return keywords
    except Exception as e:
        logger.error(f"Error in keyword_extraction: {e}")  # Ghi log lỗi
        return []  # Trả về danh sách rỗng nếu có lỗi

def isRelevant(text, keywords):
    """Kiểm tra xem văn bản có chứa từ khóa không"""
    try:
        if not text or not keywords:
            return True  # Nếu không có từ khóa hoặc văn bản, coi là liên quan
            
        text = text.lower().replace('_', ' ')  # Chuyển thành chữ thường và thay thế dấu gạch dưới
        
        # Đếm số nhóm từ khóa khớp
        matched_keywords = 0
        for words in keywords:  # Duyệt qua từng nhóm từ khóa
            if any(e for e in words if e in text):  # Nếu có bất kỳ từ nào trong nhóm xuất hiện trong văn bản
                matched_keywords += 1
        
        # Coi là liên quan nếu ít nhất một nửa nhóm từ khóa khớp
        # hoặc ít nhất một nhóm từ khóa khớp nếu có ít từ khóa
        min_matches = max(1, len(keywords) // 3)  # Số lượng khớp tối thiểu
        return matched_keywords >= min_matches
    except Exception as e:
        logger.error(f"Error in isRelevant: {e}")  # Ghi log lỗi
        return True  # Mặc định là liên quan trong trường hợp lỗi

def removeDuplicate(documents):
    """Loại bỏ văn bản trùng lặp dựa trên độ tương đồng nội dung"""
    try:
        if not documents:
            return []  # Trả về danh sách rỗng nếu không có văn bản
            
        # Tạo unigram cho mỗi văn bản
        mapUnigram = {}
        for doc in documents:
            mapUnigram[doc] = set(generateNgram(doc.lower(), 1, '_', punct_set | stopwords))

        uniqueDocs = []
        for i in range(0, len(documents)):
            check = True  # Giả định văn bản là duy nhất
            for j in range(0, len(uniqueDocs)):
                check_doc = mapUnigram[documents[i]]  # Unigram của văn bản đang kiểm tra
                exists_doc = mapUnigram[uniqueDocs[j]]  # Unigram của văn bản đã có trong danh sách duy nhất
                
                if not check_doc or not exists_doc:
                    continue  # Bỏ qua nếu không có unigram
                    
                # Tính phần trăm trùng lặp
                overlap_score = len(check_doc & exists_doc)  # Số unigram trùng lặp
                overlap_percent1 = overlap_score / len(check_doc) if check_doc else 0  # Phần trăm trùng lặp so với văn bản đang kiểm tra
                overlap_percent2 = overlap_score / len(exists_doc) if exists_doc else 0  # Phần trăm trùng lặp so với văn bản đã có
                
                if overlap_percent1 >= 0.7 or overlap_percent2 >= 0.7:  # Nếu trùng lặp ít nhất 70%
                    check = False  # Đánh dấu là trùng lặp
                    break
                    
            if check:
                uniqueDocs.append(documents[i])  # Thêm vào danh sách duy nhất nếu không trùng lặp
        
        return uniqueDocs
    except Exception as e:
        logger.error(f"Error in removeDuplicate: {e}")  # Ghi log lỗi
        return documents  # Trả về văn bản gốc trong trường hợp lỗi

def rel_ranking(question, documents):
    """Xếp hạng đoạn văn theo độ liên quan đến câu hỏi"""
    try:
        # Trả về danh sách rỗng nếu không có văn bản
        if not documents:
            logger.warning("No documents provided to rel_ranking")  # Ghi log cảnh báo
            return []
            
        logger.info(f"Ranking {len(documents)} documents for relevance")  # Ghi log thông tin
        
        # Tạo pool đa luồng
        pool = None
        try:
            pool = Pool(processes=4)  # Tạo pool với 4 tiến trình
            
            # Tạo các biến thể câu hỏi và trích xuất từ khóa
            q_variants = generateVariants(question)  # Tạo các biến thể của câu hỏi
            q_keywords = keyword_extraction(question)  # Trích xuất từ khóa từ câu hỏi
            
            logger.info(f"Extracted {len(q_keywords)} keyword groups from question")  # Ghi log thông tin
            
            # Tạo n-gram từ câu hỏi
            q_ngrams = {
                'unigram': set(generateNgram(question.lower(), 1, '_', punct_set | stopwords)),  # Tạo unigram
                'bigram': set(),  # Khởi tạo tập bigram trống
                'trigram': set(),  # Khởi tạo tập trigram trống
                'fourgram': set()  # Khởi tạo tập 4-gram trống
            }

            # Thêm n-gram từ các biến thể câu hỏi
            for q in q_variants:
                q = q.lower()  # Chuyển thành chữ thường
                # Tạo và thêm n-gram từ biến thể
                q_ngrams['bigram'] = q_ngrams['bigram'] | set(generateNgram(q, 2, '_', punct_set | stopwords))
                q_ngrams['trigram'] = q_ngrams['trigram'] | set(generateNgram(q, 3, '_', punct_set | stopwords))
                q_ngrams['fourgram'] = q_ngrams['fourgram'] | set(generateNgram(q, 4, '_', punct_set))
            
            # Lọc văn bản theo độ liên quan
            filtered_documents = [d for d in documents if isRelevant(d, q_keywords)]  # Lọc văn bản có chứa từ khóa
            
            # Nếu không có văn bản nào vượt qua bộ lọc, sử dụng tất cả văn bản
            if not filtered_documents:
                logger.warning("No documents passed relevance filter, using all documents")  # Ghi log cảnh báo
                filtered_documents = documents
                
            logger.info(f"{len(filtered_documents)} documents passed relevance filter")  # Ghi log thông tin
            
            # Tạo đoạn văn từ văn bản
            all_passages = []
            for d in filtered_documents:
                passages = generatePassages(d, 3)  # Tạo các đoạn văn có 3 câu
                all_passages.extend(passages)  # Thêm vào danh sách tất cả đoạn văn
                
            # Loại bỏ đoạn văn trùng lặp và trống
            all_passages = [p for p in all_passages if p and p.strip()]  # Lọc bỏ đoạn văn trống
            all_passages = list(set(all_passages))  # Loại bỏ trùng lặp
            
            logger.info(f"Generated {len(all_passages)} passages from documents")  # Ghi log thông tin
            
            # Lọc đoạn văn theo độ liên quan
            filtered_passages = [p for p in all_passages if isRelevant(p, q_keywords)]  # Lọc đoạn văn có chứa từ khóa
            
            # Nếu không có đoạn văn nào vượt qua bộ lọc, sử dụng tất cả đoạn văn
            if not filtered_passages:
                logger.warning("No passages passed relevance filter, using all passages")  # Ghi log cảnh báo
                filtered_passages = all_passages
                
            logger.info(f"{len(filtered_passages)} passages passed relevance filter")  # Ghi log thông tin
            
            # Tính điểm đoạn văn song song
            p_scores = []
            if filtered_passages:
                # Sử dụng pool để tính điểm song song
                p_scores = pool.map(passage_score_wrap, [(q_ngrams, p) for p in filtered_passages])
            
            # Sắp xếp đoạn văn theo điểm
            p_res = numpy.argsort([-s for s in p_scores]) if p_scores else []  # Sắp xếp theo thứ tự giảm dần
            
            # Tạo danh sách kết quả theo thứ tự liên quan
            relevantDocs = []
            for i in range(min(len(p_res), len(filtered_passages))):
                relevantDocs.append(filtered_passages[p_res[i]])  # Thêm đoạn văn theo thứ tự điểm
                
            # Loại bỏ trùng lặp
            relevantDocs = removeDuplicate(relevantDocs)  # Loại bỏ đoạn văn trùng lặp
            
            logger.info(f"Returning {len(relevantDocs)} relevant passages")  # Ghi log thông tin
            return relevantDocs  # Trả về danh sách đoạn văn đã sắp xếp
            
        except Exception as e:
            logger.error(f"Error in rel_ranking: {e}")  # Ghi log lỗi
            logger.error(traceback.format_exc())  # Ghi log stack trace
            # Trả về văn bản gốc trong trường hợp lỗi
            return documents
            
        finally:
            # Dọn dẹp pool
            if pool:
                pool.close()  # Đóng pool
                pool.join()  # Đợi tất cả tiến trình hoàn thành
                
    except Exception as e:
        logger.error(f"Error in rel_ranking outer block: {e}")  # Ghi log lỗi
        logger.error(traceback.format_exc())  # Ghi log stack trace
        return documents  # Trả về văn bản gốc trong trường hợp lỗi


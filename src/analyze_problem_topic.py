import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.cluster import KMeans, AgglomerativeClustering
import ast
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.font_manager as fm

class QuestionClassifier:
    """
    문제 텍스트를 분류하는 클래스
    
    Attributes:
        model: SentenceTransformer 모델
        embeddings: 생성된 텍스트 임베딩
        texts: 원본 텍스트 리스트
        clusters: 클러스터링 결과
    """

    def __init__(self, model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS'):
        """
        QuestionClassifier 초기화
        
        Args:
            model_name (str): 사용할 SentenceTransformer 모델 이름
        """
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.embeddings = None
        self.texts = None
        self.clusters = None
        
    def extract_questions(self, df):
        """
        DataFrame에서 문제 텍스트를 추출
        
        Args:
            df (pandas.DataFrame): 문제 데이터가 포함된 DataFrame
            
        Returns:
            list: 추출된 문제 텍스트 리스트
            list: 해당하는 클래스 레이블 리스트 (없으면 None)
        """
        questions = []
        labels = []
        for idx, row in df.iterrows():
            try:
                problem_dict = ast.literal_eval(row['problems'])
                question = problem_dict['question']
                questions.append(question)
                if 'class' in df.columns:  # class 컬럼이 있는 경우에만 레이블 추가
                    labels.append(row['class'])
            except:
                continue
        
        return questions, labels if labels else None
    
    def create_embeddings(self, questions):
        """텍스트를 임베딩 벡터로 변환"""
        print("Sample questions:")
        print(questions[:5])
        
        self.texts = questions
        self.embeddings = self.model.encode(questions)
        
        print("Embedding shape:", self.embeddings.shape)
        print("Sample embedding stats:")
        print("Mean:", np.mean(self.embeddings))
        print("Std:", np.std(self.embeddings))
        
        return self.embeddings
    
    def cluster_texts(self):
        """
        임베딩된 텍스트를 클러스터링
        
        Returns:
            numpy.ndarray: 클러스터링 결과
        """
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.embeddings)
        
        # 클러스터링 결과 확인
        print("Cluster distribution:")
        print(np.unique(clusters, return_counts=True))
        
        return clusters
    
    def visualize_clusters(self, labels):
        """
        클러스터링 결과를 시각화
        Args:
            labels: classified_train.csv의 class 컬럼 값들
        """
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'NanumGothic'  # 또는 다른 한글 폰트
        # 또는
        # font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 실제 경로로 수정 필요
        # font_prop = fm.FontProperties(fname=font_path)
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # 실제 레이블 분포 계산
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=labels,
                             cmap='Set1')
        
        # 레전드 생성
        legend_labels = {
            0: f'국어 ({label_counts.get(0, 0)}개)', 
            1: f'사회 ({label_counts.get(1, 0)}개)'
        }
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=scatter.cmap(scatter.norm(i)), 
                      label=legend_labels[i], markersize=10)
            for i in unique_labels
        ]
        
        plt.legend(handles=legend_elements)
        plt.title('문제 유형 분포')
        plt.savefig('problem_type_distribution.png')
        plt.close()
    
    def save_classification_results(self, df, output_path='after_classification.csv'):
        """
        분류 결과를 CSV 파일로 저장
        
        Args:
            df (pandas.DataFrame): 원본 데이터프레임
            output_path (str): 저장할 파일 경로
        """
        results_df = df.copy()
        
        # class 컬럼 추가
        results_df['class'] = self.clusters
        
        # 결과를 CSV 파일로 저장
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"분류 결과가 {output_path}에 저장되었습니다.")
        
        # 클러스터와 실제 레이블 간의 매핑 확인
        if hasattr(self, 'clusters'):
            cluster_label_mapping = {}
            # 클러스터 0과 1 중 어느 것이 어느 클래스에 해당하는지 확인
            print("Cluster to label mapping:")
            print(cluster_label_mapping)
    
    def prepare_training_data(self, df):
        # problems 컬럼에서 question 추출
        questions = []
        for problem in df['problems']:
            try:
                problem_dict = eval(problem)  # 문자열을 딕셔너리로 변환
                questions.append(problem_dict['question'])
            except:
                continue
            
        # 레이블 가져오기 (0: 국어, 1: 사회)
        labels = df['class'].tolist()
        
        return questions, labels
    
    def fine_tune(self, train_df, epochs=3):
        # 학습 데이터 준비
        questions, labels = self.prepare_training_data(train_df)
        print(f"학습 데이터 수: {len(questions)}")
        
        # 임베딩 생성
        embeddings = self.create_embeddings(questions)
        
        # 학습 진행
        train_examples = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = 1.0 if labels[i] == labels[j] else 0.0
                train_examples.append(InputExample(
                    texts=[questions[i], questions[j]], 
                    label=similarity
                ))
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # optimizer_params 설정
        optimizer_params = {
            'lr': 1e-5,  # 낮은 학습률
            'eps': 1e-8
        }
        
        # 학습 진행
        print("파인튜닝 시작...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=50,
            optimizer_params=optimizer_params,
            show_progress_bar=True
        )
        
        print("파인튜닝 완료")
    
    def train_classifier(self, embeddings, labels, epochs):
        # 학습 데이터 준비
        train_examples = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # 같은 클래스면 1, 다른 클래스면 0
                similarity = 1.0 if labels[i] == labels[j] else 0.0
                
                train_examples.append(InputExample(
                    texts=[embeddings[i], embeddings[j]], 
                    label=similarity
                ))
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Cosine Similarity 손실 함수 사용
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # 학습 진행
        print("파인튜닝 시작...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            show_progress_bar=True
        )
        
        # 파인튜닝된 모델 저장
        self.model.save('fine_tuned_model')
        print("파인튜닝 완료 및 모델 저장")
    
    def predict_class(self, new_text, df):
        """새로운 질문의 클래스 예측"""
        # 새 질문의 임베딩
        new_embedding = self.model.encode([new_text])
        
        # 기존 질문들과 비교
        similarities = []
        questions, labels = self.extract_questions(df)
        
        for q, label in zip(questions, labels):
            embedding = self.model.encode([q])
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(new_embedding), 
                torch.tensor(embedding)
            )
            similarities.append((similarity.item(), label))
        
        # 가장 유사한 상위 5개 질문의 클래스로 예측
        similarities.sort(reverse=True)
        top_k = similarities[:5]
        predicted_class = max(set([c for _, c in top_k]), 
                            key=lambda x: sum(1 for s, c in top_k if c == x))
        
        return predicted_class
    
    def evaluate_clustering(self, true_labels):
        """
        클러스터링 성능 평가
        
        Args:
            true_labels (list): 실제 클래스 레이블
        """
        # 클러스터링 결과와 실제 레이블 비교
        ari = adjusted_rand_score(true_labels, self.clusters)
        nmi = normalized_mutual_info_score(true_labels, self.clusters)
        
        print("러스터링 성능 평가:")
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Normalized Mutual Information: {nmi:.3f}")

def main():

    
    # 학습용 데이터 로드
    train_df = pd.read_csv('/data/ephemeral/home/ksw/train_augmented_google_ja_1loop.csv')
    
    # 분류기 초기화 및 학습
    classifier = QuestionClassifier()
    print("파인튜닝 시작...")
    classifier.fine_tune(train_df, epochs=2)
    print("파인튜닝 완료")
    

    # 파인튜닝된 모델 로드
    #classifier = QuestionClassifier(model_name='/data/ephemeral/home/ksw/fine_tuned_model')
    
    # 실제 분류할 데이터 로드
    target_df = pd.read_csv('/data/ephemeral/home/ksw/level2-nlp-generationfornlp-nlp-07-lv3/data/train.csv')
    
    # 분류 대상 데이터에서 질문 추출
    questions, _ = classifier.extract_questions(target_df)
    print(f"추출된 질문 수: {len(questions)}")
    
    # 임베딩 생성 및 클러스터링
    classifier.create_embeddings(questions)
    classifier.clusters = classifier.cluster_texts()
    
    # 결과 저장
    output_path = 'after_classification.csv'
    classifier.save_classification_results(target_df, output_path)
    
    # 클러스터링 결과를 직접 시각화
    if classifier.clusters is not None:
        classifier.visualize_clusters(classifier.clusters)

if __name__ == "__main__":
    main()
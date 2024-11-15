import pandas as pd

# CSV 파일 로드
df = pd.read_csv('./data/train.csv')

# 삭제할 숫자 리스트 선언
delete_numbers = [426, 498]  # 예: 삭제하고자 하는 id 번호 (뒤의 숫자만 봄)

# 정확히 해당 숫자로 끝나는 경우만 행 삭제,  예 : 426은 426만 삭제 , 1426, 2426등 무관
pattern = r'(^|[^0-9])(' + '|'.join(str(num) + r'$' for num in delete_numbers) + r')'
df_filtered = df[~df['id'].str.contains(pattern, regex=True)]

# 필터링된 결과 확인
print(df_filtered)

# 필터링된 데이터 저장
df_filtered.to_csv('./data/filtered_file.csv', index=False)
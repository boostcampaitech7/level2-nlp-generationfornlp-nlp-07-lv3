import streamlit as st
import pandas as pd
from Augmentor_by_BT import BackTranslator

def main():
    st.title("Back Translation 데이터 증강기")
    
    # 세션 스테이트 초기화
    if 'previous_results' not in st.session_state:
        st.session_state.previous_results = []
    
    # 이전 결과물 표시
    if st.session_state.previous_results:
        st.subheader("이전 결과물")
        for idx, result in enumerate(st.session_state.previous_results):
            with st.expander(f"증강 결과 #{idx+1}"):
                st.write("증강 조건:")
                st.write(f"- 선택된 ID 컬럼: {result['id_column']}")
                st.write(f"- 증강된 컬럼: {', '.join(result['augment_columns'])}")
                st.write(f"- 번역기: {result['translator_type']}")
                st.write(f"- 중간 번역 언어: {result['target_lang']}")
                st.write(f"- 배치 크기: {result['batch_size']}")
                st.write(f"- 역번역 반복 횟수: {result['loop_count']}")
                
                # 다운로드 버튼
                st.download_button(
                    label=f"결과 #{idx+1} 다운로드",
                    data=result['data'],
                    file_name=f"augmented_data_{idx+1}.csv",
                    mime="text/csv",
                    help=f"""증강 조건:
                    ID 컬럼: {result['id_column']}
                    증강 컬럼: {', '.join(result['augment_columns'])}
                    번역기: {result['translator_type']}
                    중간 언어: {result['target_lang']}
                    배치 크기: {result['batch_size']}
                    반복 횟수: {result['loop_count']}"""
                )
    
    # CSS로 바깥쪽 스크롤바 제거
    st.markdown("""
        <style>
            [data-testid="stVerticalBlock"] {
                overflow: hidden;
            }
        </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    
    if uploaded_file is not None:
        # 데이터 로드 및 전체 미리보기
        df = pd.read_csv(uploaded_file)
        st.subheader("원본 데이터 미리보기")
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        st.write(f"총 {len(df)}개 행")
        
        # 컬럼 선택을 위한 체크박스들
        st.subheader("컬럼 설정")
        col1, col2 = st.columns([1, 3])  # 1:3 비율로 컬럼 분할
        
        with col1:
            st.write("ID 역할 컬럼 선택 (1개만)")
            id_columns = {}
            for col in df.columns:
                id_columns[col] = st.checkbox(f"{col}", key=f"id_{col}")
            
            # ID 컬럼이 1개만 선택되었는지 확인
            selected_id_columns = [col for col, selected in id_columns.items() if selected]
            if len(selected_id_columns) > 1:
                st.error("ID 컬럼은 1개만 선택 가능합니다.")
                return
            
            # ID 컬럼이 선택되었을 때만 증가 여부 체크박스 표시
            if selected_id_columns:
                increment_id = st.checkbox(
                    "ID 자동 증가",
                    value=True,
                    help="체크 해제 시 원본 ID를 그대로 복제합니다."
                )
        
        with col2:
            st.write("증강할 컬럼 선택 (1개 이상)")
            augment_columns = {}
            for col in df.columns:
                # ID로 선택된 컬럼은 비활성화하고, ID가 선택되지 않았으면 모든 컬럼 비활성
                disabled = col in selected_id_columns or not selected_id_columns
                augment_columns[col] = st.checkbox(
                    f"{col}",
                    key=f"augment_{col}",
                    disabled=disabled
                )
        
        # ID 컬럼이 선택되지 않았을 때 안내 메시지
        if not selected_id_columns:
            st.warning("먼저 왼쪽에서 ID 역할 컬럼을 선택해주세요.")
            return
            
        # 증강할 컬럼이 1개 이상 선택되었는지 확인
        selected_augment_columns = [col for col, selected in augment_columns.items() if selected]
        if not selected_augment_columns:
            st.warning("증강할 컬럼을 1개 이상 선택해주세요.")
            return
        
        # 나머지 설정 파라미터들
        st.subheader("증강 설정")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            translator_type = st.selectbox(
                "번역기 선택",
                ["google", "deepl"],
                help="Google: 무료/일일 제한 있음, DeepL: API 키 필요"
            )
        
        with col2:
            target_lang = st.selectbox(
                "중간 번역 언어",
                ["en", "ja"],
                help="영어(en)나 일본어(ja) 중 선택"
            )
            
        with col3:
            batch_size = st.number_input(
                "배치 크기",
                min_value=1,
                max_value=32,
                value=16,
                help="한 번에 처리할 데이터 수 (클수록 빠르지만 메모리 많이 사용)"
            )
            
        with col4:
            loop_count = st.number_input(
                "역번역 반복 횟수",
                min_value=1,
                max_value=5,
                value=1
            )
        
        # DeepL API 키 입력 (선택적)
        if translator_type == "deepl":
            deepl_api_key = st.text_input("DeepL API 키를 입력하세요", type="password")
        else:
            deepl_api_key = None
            
        # 증강 실행 버튼
        if st.button("데이터 증강 시작"):
            try:
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                with st.spinner("데이터 증강 중..."):
                    bt = BackTranslator(
                        type=translator_type,
                        loop=loop_count,
                        lang=target_lang,
                        batch_size=batch_size,
                        deepl_api_key=deepl_api_key
                    )
                    
                    # 증강된 데이터 생성
                    augmented_df = bt.augment(
                        df=df,
                        augment_columns=selected_augment_columns,
                        id_column=selected_id_columns[0],
                        increment_id=increment_id
                    )
                    
                    # 원본과 증강 데이터 합치기
                    final_df = pd.concat([df, augmented_df], ignore_index=True)
                    
                    # 결과 표시
                    status_text.text("완료!")
                    st.success("데이터 증강 완료!")
                    
                    # 선택된 컬럼만 포함하는 DataFrame 생성
                    display_columns = selected_augment_columns + [selected_id_columns[0]]  # ID 컬럼 추가
                    df_display = df[display_columns]
                    augmented_df_display = augmented_df[display_columns]
                    
                    # 비교 표시
                    st.subheader("데이터 비교")
                    comparison_col1, comparison_col2 = st.columns(2)
                    
                    with comparison_col1:
                        st.write("원본 데이터")
                        st.dataframe(df_display, use_container_width=True)
                        st.write(f"총 {len(df)}개 행")
                        
                        # 원본 데이터 통계
                        st.write("선택된 컬럼의 원본 데이터 통계")
                        for col in selected_augment_columns:
                            if col != 'problems':  # problems 컬럼 제외
                                st.write(f"- {col} 평균 길이: {df[col].str.len().mean():.1f}")
                    
                    with comparison_col2:
                        st.write("증강된 데이터")
                        st.dataframe(
                            augmented_df,  # 증강된 데이터만 표시
                            use_container_width=True
                        )
                        st.write(f"총 {len(augmented_df)}개 행")
                    
                    # 상세 비교 (선택된 컬럼만)
                    st.subheader("상세 비교")
                    for col in selected_augment_columns:
                        st.write(f"\n### {col} 컬럼 비교")
                        compare_col1, compare_col2 = st.columns(2)
                        
                        with compare_col1:
                            st.write("원본:")
                            if col == 'problems':
                                problems_data = df[col].apply(eval)
                                questions = problems_data.apply(lambda x: x['question'])
                                st.dataframe(questions, use_container_width=True)
                            else:
                                st.dataframe(df[col], use_container_width=True)
                        
                        with compare_col2:
                            st.write("증강:")
                            if col == 'problems':
                                problems_data = augmented_df[col].apply(eval)  # .iloc[len(df):] 제거
                                questions = problems_data.apply(lambda x: x['question'])
                                st.dataframe(questions, use_container_width=True)
                            else:
                                st.dataframe(
                                    augmented_df[col],  # .iloc[len(df):] 제거
                                    use_container_width=True
                                )
                    
                    # 결과 저장
                    result_info = {
                        'id_column': selected_id_columns[0],
                        'augment_columns': selected_augment_columns,
                        'translator_type': translator_type,
                        'target_lang': target_lang,
                        'batch_size': batch_size,
                        'loop_count': loop_count,
                        'data': final_df.to_csv(index=False)
                    }
                    
                    # 최대 5개까지만 저장
                    if len(st.session_state.previous_results) >= 5:
                        st.session_state.previous_results.pop(0)
                    st.session_state.previous_results.append(result_info)
                    
                    # 현재 결과 다운로드 버튼
                    st.download_button(
                        label="증강된 데이터 다운로드 (원본+증강)",
                        data=result_info['data'],
                        file_name="augmented_data.csv",
                        mime="text/csv",
                        help=f"""증강 조건:
                        ID 컬럼: {result_info['id_column']}
                        증강 컬럼: {', '.join(result_info['augment_columns'])}
                        번역기: {result_info['translator_type']}
                        중간 언어: {result_info['target_lang']}
                        배치 크기: {result_info['batch_size']}
                        반복 횟수: {result_info['loop_count']}"""
                    )
                    
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()

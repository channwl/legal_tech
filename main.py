import streamlit as st
import PyPDF2
import openai
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pdfminer.high_level import extract_text
import re
import os
import subprocess
from io import BytesIO
import os

import streamlit as st

# API 키 불러오기
api_key = st.secrets.get("general", {}).get("API_KEY", None)

if api_key is None:
    st.error("API 키가 설정되지 않았습니다. secrets.toml 또는 Streamlit Cloud Secrets에서 설정하세요.")
else:
    st.success("API 키가 정상적으로 로드되었습니다.")




def extract_and_clean_text(file):
    criteria = extract_text(file).strip()
    return criteria.strip()

def extract_text_from_pdf(file):
    structured_sections = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        structured_sections += page.extract_text() + "\n"
    return structured_sections

def parse_scores(result_text, question_count):
    """Parse scores based on the number of questions"""
    scores = {}
    
    if question_count == 1:
        # Single question format
        total_match = re.search(r"총점\s*:\s*(\d+)", result_text)
        if total_match:
            scores["총점"] = int(total_match.group(1))
    else:
        # Multiple questions format
        question_scores = re.finditer(r"문제\s*(\d+-\d+)\s*총점\s*:\s*(\d+)", result_text)
        for match in question_scores:
            question_num = match.group(1)
            score = int(match.group(2))
            scores[f"문제{question_num}"] = score
    
    return scores



def get_grading_prompt(question_count):
    """Return appropriate prompts based on number of questions"""
    system_prompt = """
    당신은 법학 서술형 답안을 채점하는 엄격하고 공정한 채점관입니다.
    모든 채점 기준을 세밀하게 검토하고, 채점 기준에 따른 충족 여부를 명확하게 판단하세요.
    - 문제가 모호할 경우 항상 보수적으로 판단하고, 학생이 명확히 설명하지 못한 부분은 감점하세요.
    - 채점기준에 나온 '제n조'가 명시 되어있지않으면 감점하세요.
    - 채점 기준을 하나씩 다 나눠서 채점해주세요.
    - 점수 부여 시 근거를 명확하게 확인해주세요.
    """

    if question_count == 1:
        user_prompt_template = """
        채점 기준:
        {guideline}

        학생 답안:
        {answer}

        위 정보를 바탕으로:
        1. 채점 기준에 따라 학생 답안이 얼마나 충족되었는지 평가하세요.
        2. 채점기준에 나온 '제n조'가 명시 되어있지않으면 감점하세요.
        3. 채점 기준을 하나씩 다 나눠서 채점해주세요.
        4. 점수를 부여하고 근거를 명확히 설명해주세요.
        5. 점수는 정수로 나타내주세요.

        출력 형식은 아래와 같습니다:
        문제 1
        - 근거 :
        - 문제 1 총점 : [숫자]
        """
    else:
        user_prompt_template = """
        채점 기준:
        {guideline}

        학생 답안:
        {answer}

        위 정보를 바탕으로:
        1. 각 문제(예: 2-1, 2-2)별로 나눠서 채점해주세요.
        2. 각 채점 기준에 따라 학생 답안이 얼마나 충족되었는지 평가하세요.
        3. 채점기준에 나온 '제n조'가 명시 되어있지않으면 감점하세요.
        4. 채점 기준을 하나씩 다 나눠서 채점해주세요.
        5. 각 문제별 점수를 명시해주세요.
        6. 점수는 정수로 나타내주세요.

        출력 형식은 아래와 같습니다:
        문제 2-1
        - 근거 :
        - 문제 2-1 총점 : [숫자]

        문제 2-2
        - 근거 :
        - 문제 2-2 총점 : [숫자]
        """
    
    return system_prompt, user_prompt_template

def grade_with_openai(guideline, answer, question_count):
    """Grade answers using OpenAI API with appropriate prompts"""
    system_prompt, user_prompt_template = get_grading_prompt(question_count)

    # Format user prompt
    user_prompt = user_prompt_template.format(
        guideline=guideline,
        answer=answer
    )

    # API call
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Using gpt-4o as requested
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
            temperature=0,
    )

    # Return the response content
    return response["choices"][0]["message"]["content"].strip()

def clear_uploaded_files():
    """Clear all uploaded files and reset the session state."""
    st.session_state.results = []
    st.session_state.graph_data = {}
    if 'criteria_file' in st.session_state:
        del st.session_state['criteria_file']
    if 'answer_files' in st.session_state:
        del st.session_state['answer_files']

    st.write('<meta http-equiv="refresh" content="0; url=/" />', unsafe_allow_html=True)

def merge_uploaded_csvs(uploaded_files):
    """CSV 파일을 학생번호 기준으로 병합하고 총점을 계산하는 함수"""
    dataframes = [pd.read_csv(file) for file in uploaded_files]

    # 학생번호 기준으로 병합
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.merge(df, on="학생번호", how="outer")

    # NaN 값을 0으로 변환 후 총점 계산
    score_columns = [col for col in merged_df.columns if col.startswith("문제")]
    merged_df["총점"] = merged_df[score_columns].fillna(0).sum(axis=1)

    # 총점을 맨 왼쪽으로 이동
    column_order = ["총점", "학생번호"] + score_columns
    merged_df = merged_df[column_order]

    return merged_df

def convert_df_to_csv(df):
    """DataFrame을 CSV 파일로 변환"""
    output = BytesIO()
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return output.getvalue()

def main():
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_icon="⚖️",
        page_title="법률 채점 프로그램 | FELT"
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("⚖️ 법률 채점 프로그램 | FELT")

        st.sidebar.title("📂 파일 업로드")

        if st.button("🗑️ 새로운 문제 채점", type="secondary"):
            clear_uploaded_files()
        st.text("새로운 채점이 진행될때 꼭 클릭해주세요!")

        question_count = st.sidebar.radio(
            "문제 개수를 선택하세요:",
            options=[1, 2],
            format_func=lambda x: f"{x}문제",
            index=0
        )

        criteria_file = st.sidebar.file_uploader("채점 기준 PDF 파일을 업로드하세요", type=["pdf"], key="criteria_file")
        answer_files = st.sidebar.file_uploader("학생 답안 PDF 파일을 여러 개 업로드하세요", type=["pdf"], accept_multiple_files=True, key="answer_files")

        if "results" not in st.session_state:
            st.session_state.results = []
            st.session_state.graph_data = {}

        if st.sidebar.button("✅ 채점 시작"):
            st.session_state.results = []
            st.session_state.graph_data = {}

            if criteria_file is None:
                st.sidebar.error("채점 기준 파일을 업로드해주세요.")
                return

            if not answer_files:
                st.sidebar.error("학생 답안 파일을 업로드해주세요.")
                return

            with st.spinner("채점 기준을 추출 중입니다..."):
                criteria_text = extract_and_clean_text(criteria_file)

            results = []
            graph_data = {}
            question_scores = {}

            for i, file in enumerate(answer_files):
                with st.spinner(f"학생 답안 {i + 1} 채점 중입니다..."):
                    answer_text = extract_text_from_pdf(file)
                    result = grade_with_openai(criteria_text, answer_text, question_count)
                    results.append((file.name, result))

                    # Parse scores for graph data
                    scores = parse_scores(result, question_count)
                    for question, score in scores.items():
                        if question not in graph_data:
                            graph_data[question] = []
                        graph_data[question].append(score)

                        if question not in question_scores:
                            question_scores[question] = []
                        question_scores[question].append(score)

            st.session_state.results = results
            st.session_state.graph_data = graph_data

            st.subheader("채점 결과")
            csv_data = []

            for file_name, result in results:
                st.write(f"**학생 답안 파일명: {file_name}**")
                st.text(result)
                st.write("---")

                file_name = file_name.replace('.pdf', '')
                scores = parse_scores(result, question_count)
                row_data = {"학생번호": file_name}
                row_data.update(scores)
                csv_data.append(row_data)

            if csv_data:
                csv_df = pd.DataFrame(csv_data)
                if question_count == 2 and "총점" in csv_df.columns:
                    csv_df = csv_df.drop(columns=["총점"])
                csv_file = "grading_results.csv"
                csv_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
                st.sidebar.download_button(
                    label="📥 채점 결과 CSV 다운로드",
                    data=open(csv_file, "rb"),
                    file_name="grading_results.csv",
                    mime="text/csv"
                )
                
        st.sidebar.subheader("📂 CSV 파일 업로드 및 병합")
        uploaded_csvs = st.sidebar.file_uploader("채점 결과 CSV 파일을 업로드하세요", type=["csv"], accept_multiple_files=True)

        if uploaded_csvs:
            merged_df = merge_uploaded_csvs(uploaded_csvs)
            st.subheader("📊 병합된 채점 결과")
            st.write(merged_df)

            st.sidebar.download_button(
                label="📥 병합된 채점 결과 CSV 다운로드",
                data=convert_df_to_csv(merged_df),
                file_name="merged_grading_results.csv",
                mime="text/csv",
            )
        else:
            st.sidebar.info("CSV 파일을 업로드하면 학생번호 기준으로 데이터를 병합할 수 있습니다.")

    with col2:
        st.header("📊 채점 결과")

        if st.session_state.results:
            graph_data = st.session_state.graph_data

            for question, scores in graph_data.items():
                st.subheader(f"{question} 분포")

                fig, ax = plt.subplots(figsize=(8, 6))
                score_counts = pd.Series(scores).value_counts().sort_index()
                ax.bar(score_counts.index, score_counts.values,  # 수정: index를 그대로 사용
                    color="skyblue", edgecolor="black")
                ax.set_xlabel("Score")
                ax.set_ylabel("Number of students")
                ax.set_title(f"Distribution")
                ax.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig)

                # Display statistics for each question
                st.write(f"**{question} 통계 정보:**")
                st.write(f"- 최고 점수: {max(scores)}")
                st.write(f"- 최저 점수: {min(scores)}")
                st.write(f"- 평균 점수: {np.mean(scores):.2f}")

        else:
            st.info("채점 결과가 아직 없습니다. PDF 파일을 업로드하고 채점을 시작하세요.")

if __name__ == "__main__":
    main()

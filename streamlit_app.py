# streamlit_app.py
"""
Streamlit 앱: 해수면온도(SST) vs 폭염일수 대시보드 (한국어 UI)

- 공개 데이터 URL 접근 부분 제거, 예시 데이터 기반으로 시각화
- 사용자 입력 텍스트 기반 합성 데이터 포함
- 기능 개선:
  - 다중 변수 상관관계 분석 및 시각화 추가 (폭염일수, 해수면온도, 수면시간)
  - 상관계수 계산 방식 선택 기능 (피어슨, 스피어만)
  - 사용자 맞춤형 분석 의견 입력 및 대시보드에 표시 기능
  - 특정 연도 데이터 강조 및 가정된 지역명 입력 기능 추가
  - 상관관계 히트맵 시각화 추가
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 설정 ----------
st.set_page_config(page_title="폭염·해수온 대시보드", layout="wide", initial_sidebar_state="expanded")

# Pretendard 폰트 적용 시도 (주석 처리)
# Pretendard 폰트는 로컬 환경에서만 작동하므로, 웹 앱 배포 시 문제가 발생할 수 있어 주석 처리했습니다.
# PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
# _custom_css = f"""
# <style>
# @font-face {{
#   font-family: 'PretendardCustom';
#   src: url('{PRETENDARD_PATH}');
# }}
# html, body, [class*="css"] {{
#   font-family: PretendardCustom, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
# }}
# </style>
# """
# st.markdown(_custom_css, unsafe_allow_html=True)
# try:
#     import matplotlib.font_manager as fm
#     fm.fontManager.addfont(PRETENDARD_PATH)
#     plt.rcParams['font.family'] = 'PretendardCustom'
# except Exception:
#     pass

# ---------- 유틸리티 ----------
@st.cache_data
def today_local_date():
    tz = pytz.timezone("Asia/Seoul")
    return dt.datetime.now(tz).date()

TODAY = today_local_date()

def remove_future_dates(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.date <= TODAY]
    return df

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------- 공개 데이터 예시 (기존 코드) ----------
st.header("공개 데이터 기반 대시보드 (예시 데이터 사용)")

public_data_notice = st.empty()
public_warning = True

years = np.arange(2005, 2025)
dates = pd.to_datetime([f"{y}-07-01" for y in years])
sst_values = 20.0 + (years - 2005) * 0.02 + np.random.normal(0, 0.08, len(years))
public_sst_df = pd.DataFrame({'date': dates, 'sst': sst_values})

col1, col2 = st.columns([3,1])
with col1:
    fig = px.line(public_sst_df, x='date', y='sst', title='한반도 주변 해수면 온도(예시 데이터)',
                  labels={'date':'연도', 'sst':'해수면 온도 (℃)'})
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.markdown("**데이터 출처(예시)**")
    st.write("- NOAA ERSST v5 (예시 URL).")
    st.write("- NOAA OISST (대체 가능).")
    st.write("- 기상청 폭염일수 포털 예시 URL")
    st.write("**알림:** 공개 데이터 접근이 불완전하여 예시 데이터 사용 중입니다.")
    st.download_button("해수면온도 CSV 다운로드", data=df_to_csv_bytes(public_sst_df), file_name="public_sst_preprocessed.csv", mime="text/csv")

# ---------- 사용자 입력 기반 대시보드 ----------
st.header("사용자 입력 기반 대시보드 — 입력 텍스트 분석 결과")

@st.cache_data
def synthesize_from_text():
    years = np.arange(2005, 2025)
    sst = 20.0 + (years - 2005) * 0.02 + np.random.normal(0, 0.05, len(years))
    heatdays = 5 + (years - 2005) * (13 / 20) + np.random.normal(0, 1.5, len(years))
    heatdays = np.clip(heatdays.round(1), 0, None)
    sleep_hours = 8.5 - (years - 2005) * (0.3 / 20) + np.random.normal(0, 0.05, len(years))
    df = pd.DataFrame({
        'year': years,
        'date': pd.to_datetime([f"{y}-07-01" for y in years]),
        'sst': sst,
        'heatwave_days': heatdays,
        'avg_sleep_hours': sleep_hours
    })
    df = remove_future_dates(df, 'date')
    return df

user_df = synthesize_from_text()

# Side Bar Filters and Options
st.sidebar.header("필터 · 옵션")
yr_min = int(user_df['year'].min())
yr_max = int(user_df['year'].max())
year_range = st.sidebar.slider("연도 범위 선택", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), step=1)
smoothing_window = st.sidebar.selectbox("시계열 스무딩(이동평균) 기간(년)", options=[1,3,5], index=1)
show_trend = st.sidebar.checkbox("추세선 표시", value=True)
corr_method = st.sidebar.selectbox("상관계수 계산 방식", options=['피어슨', '스피어만'])
highlight_years = st.sidebar.multiselect("강조할 연도 선택", options=list(range(yr_min, yr_max + 1)))
assumed_region = st.sidebar.text_input("분석에 가정할 지역명 입력", "한반도")
st.sidebar.markdown("---")
st.sidebar.subheader("나만의 분석 의견")
user_conclusion = st.sidebar.text_area("분석 내용이나 결론을 작성하세요", "이 데이터에 따르면, 해수면 온도가 상승할수록 폭염일수가 증가하고 청소년의 평균 수면 시간은 감소하는 경향을 보입니다. 기후 변화가 우리 삶에 미치는 영향을 알 수 있습니다.")

df_vis = user_df[(user_df['year'] >= year_range[0]) & (user_df['year'] <= year_range[1])].copy()
df_vis['sst_smooth'] = df_vis['sst'].rolling(smoothing_window, center=True, min_periods=1).mean()
df_vis['heat_smooth'] = df_vis['heatwave_days'].rolling(smoothing_window, center=True, min_periods=1).mean()
df_vis['sleep_smooth'] = df_vis['avg_sleep_hours'].rolling(smoothing_window, center=True, min_periods=1).mean()

st.subheader("요약 시각화")
# Create a single figure with three y-axes for a comprehensive view.
fig1 = go.Figure()

# Add SST line plot
fig1.add_trace(go.Scatter(x=df_vis['year'], y=df_vis['sst_smooth'], name='해수면 온도 (℃)', mode='lines'))
# Add Heatwave Days bar plot
fig1.add_trace(go.Bar(x=df_vis['year'], y=df_vis['heat_smooth'], name='폭염일수 (일)', yaxis='y2', opacity=0.5))
# Add Sleep Hours line plot
fig1.add_trace(go.Scatter(x=df_vis['year'], y=df_vis['sleep_smooth'], name='평균 수면시간 (시간)', yaxis='y3', mode='lines'))

# Highlight selected years
for year in highlight_years:
    highlight_data = df_vis[df_vis['year'] == year]
    if not highlight_data.empty:
        fig1.add_trace(go.Scatter(
            x=[highlight_data['year'].iloc[0]], 
            y=[highlight_data['sst_smooth'].iloc[0]],
            mode='markers+text',
            text=[str(year)],
            textposition="top center",
            name=f'{year}년 강조',
            marker=dict(size=10, color='red', symbol='star')
        ))

fig1.update_layout(
    title=f'{assumed_region} 해수면 온도, 폭염일수, 평균 수면시간 추이',
    xaxis=dict(title='연도'),
    yaxis=dict(title="해수면 온도 (℃)", showgrid=False, domain=[0, 1]),
    yaxis2=dict(title="폭염일수 (일)", overlaying='y', side='right', showgrid=False),
    yaxis3=dict(title="평균 수면시간 (시간)", overlaying='y', side='right', position=0.95),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("상관관계 분석")
p1, p2, p3 = st.columns(3)
with p1:
    corr_val = df_vis['sst'].corr(df_vis['heatwave_days'], method='pearson' if corr_method == '피어슨' else 'spearman')
    st.metric(f"SST vs 폭염일수 상관계수 ({corr_method})", f"{corr_val:.2f}")
    
with p2:
    corr_val_heat_sleep = df_vis['heatwave_days'].corr(df_vis['avg_sleep_hours'], method='pearson' if corr_method == '피어슨' else 'spearman')
    st.metric(f"폭염일수 vs 수면시간 상관계수 ({corr_method})", f"{corr_val_heat_sleep:.2f}")

with p3:
    corr_val_sst_sleep = df_vis['sst'].corr(df_vis['avg_sleep_hours'], method='pearson' if corr_method == '피어슨' else 'spearman')
    st.metric(f"SST vs 수면시간 상관계수 ({corr_method})", f"{corr_val_sst_sleep:.2f}")

st.write("설명: 폭염일수-SST는 양(+)의 상관관계, 수면시간-폭염일수는 음(-)의 상관관계를 입력 텍스트 기반 합성 데이터로 재현함.")

st.subheader("주요 변수 산점도 분석")
fig2 = px.scatter(df_vis, x='sst', y='heatwave_days', color='avg_sleep_hours',
                  trendline='ols' if show_trend else None,
                  labels={'sst':'해수면 온도 (℃)', 'heatwave_days':'연간 폭염일수 (일)', 'avg_sleep_hours': '평균 수면시간 (시간)'},
                  hover_data=['year'],
                  title=f'{assumed_region} 해수면 온도 vs 폭염일수 vs 수면시간',
                  color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig2, use_container_width=True)

# 상관관계 히트맵 추가
st.subheader("변수 간 상관관계 히트맵")
corr_matrix = df_vis[['sst', 'heatwave_days', 'avg_sleep_hours']].corr(method='pearson' if corr_method == '피어슨' else 'spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f'변수 간 상관관계 ({corr_method})')
st.pyplot(plt)

st.subheader("전처리된 표 (다운로드 가능)")
st.dataframe(df_vis[['year','date','sst','heatwave_days','avg_sleep_hours']].reset_index(drop=True), use_container_width=True)
st.download_button("전처리된 표 CSV 다운로드", data=df_to_csv_bytes(df_vis[['year','date','sst','heatwave_days','avg_sleep_hours']]), file_name="user_input_synthesized.csv", mime="text/csv")

st.header("간단 결론 및 권고 (프롬프트 기반)")
st.markdown(
f"""
### 나만의 분석 의견

{user_conclusion}

### 데이터 기반 결론 및 권고

- **핵심 포인트**: {assumed_region} 주변 해수면 온도 상승은 내륙 폭염일수 증가와 양(+)의 상관관계를 보이며, 이는 청소년의 평균 수면시간에 부정적 영향을 미칠 가능성이 있습니다.
- **주요 시사점**: 기후 변화는 환경 문제를 넘어 우리 삶의 질과 건강에 직접적인 영향을 미치고 있습니다. 특히, 폭염은 청소년의 수면 패턴을 교란하여 학업 성취 및 신체·정신 건강에 해로운 영향을 줄 수 있습니다.
- **권고사항**:
  1. 폭염 시 야간 냉방 접근성 확대 및 학교 내 휴식환경 개선
  2. 기후 변화의 건강 영향에 대한 학생 교육 강화
  3. 해양 및 기후 모니터링 강화, 그리고 장기적 온실가스 감축 정책 마련
"""
)

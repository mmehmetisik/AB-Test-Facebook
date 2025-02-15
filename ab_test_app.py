# Önceki tüm importlar aynen kalacak
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="A/B Test Eğitim ve Analiz Aracı", layout="wide", initial_sidebar_state="expanded")

# Stil ve renk tanımlamaları
MAIN_COLOR = "#2E86C1"
SUCCESS_COLOR = "#28B463"
WARNING_COLOR = "#F1C40F"
ERROR_COLOR = "#E74C3C"

# Custom CSS - Ana başlık büyütüldü
st.markdown("""
    <style>
    .main-title {
        font-size: 60px !important;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin: 40px 0;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #2E86C1;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        padding: 20px;
        border-radius: 5px;
        background-color: #EBF5FB;
        margin-bottom: 20px;
    }
    .warning-box {
        padding: 20px;
        border-radius: 5px;
        background-color: #FEF9E7;
        margin-bottom: 20px;
    }
    .concept-box {
        padding: 15px;
        border-left: 3px solid #2E86C1;
        background-color: #F8F9F9;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data(uploaded_file):
    try:
        control_df = pd.read_excel(uploaded_file, sheet_name="Control Group")
        test_df = pd.read_excel(uploaded_file, sheet_name="Test Group")
        return control_df, test_df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None, None

def show_welcome():
    st.markdown('<p class="main-title">🎯 A/B Test Analiz ve Öğrenme Platformu</p>', unsafe_allow_html=True)

    # Uyarı notu ekle
    st.warning("""
        ⚠️ **Önemli Not:** 
        
        Bu uygulama, GitHub reposunda verilen 'ab_testing.xlsx' dosyası için özelleştirilmiş ve öğrenme amacıyla yapılmıştır. 
        Farklı bir veri seti ile uyumlu çalışmayabilir.
    """)

    st.markdown("""
    ### 📚 Bu Platform Nedir?

    Bu platform, A/B testlerini anlamanıza ve uygulamanıza yardımcı olmak için tasarlanmış interaktif bir eğitim ve analiz aracıdır.
    """)

    with st.expander("🎯 A/B Testi Temel Kavramlar", expanded=True):
        st.markdown("""
        #### 1. İstatistiksel Kavramlar
        - **Hipotez Testi**: Veri tabanlı karar verme süreci
        - **P-değeri**: İstatistiksel anlamlılık ölçüsü
        - **Güven Aralığı**: Tahmin aralığı (genellikle %95)
        - **Tip I Hata (α)**: Yanlış pozitif (genellikle 0.05)
        - **Tip II Hata (β)**: Yanlış negatif

        #### 2. Test Türleri
        - **Parametrik Testler**: Normal dağılım varsayımı gerektirir
        - **Non-parametrik Testler**: Dağılım varsayımı gerektirmez
        - **A/B/n Testleri**: İkiden fazla varyant testi

        #### 3. Örneklem ve Güç
        - **Örneklem Büyüklüğü**: Test için gereken minimum veri sayısı
        - **İstatistiksel Güç**: Gerçek farkı tespit etme olasılığı
        - **Etki Büyüklüğü**: Farkın pratik anlamlılığı
        """)

    with st.expander("📊 A/B Testi Uygulama Alanları", expanded=True):
        st.markdown("""
        #### 1. E-ticaret
        - **Fiyatlandırma Stratejileri**
          - Farklı fiyat noktaları testi
          - İndirim oranları karşılaştırması
        - **Ürün Sayfası Optimizasyonu**
          - Görsellerin etkisi
          - Açıklama formatları

        #### 2. Dijital Pazarlama
        - **Reklam Performansı**
          - Farklı reklam metinleri
          - Görsel varyasyonları
        - **Landing Page Optimizasyonu**
          - CTA buton renkleri
          - Sayfa düzeni değişiklikleri

        #### 3. UX/UI Tasarımı
        - **Navigasyon Değişiklikleri**
          - Menü yapısı
          - Buton yerleşimi
        - **İçerik Stratejisi**
          - Başlık formatları
          - İçerik uzunluğu
        """)

    with st.expander("⚠️ Sık Yapılan Hatalar ve Öneriler", expanded=True):
        st.markdown("""
        #### 1. Test Tasarım Hataları
        - Çok fazla varyant test etme
        - Yetersiz örneklem büyüklüğü
        - Test süresini erkenden sonlandırma

        #### 2. Analiz Hataları
        - Yanlış test türü seçimi
        - Çoklu test hatası
        - Segment analizi eksikliği

        #### 3. Yorumlama Hataları
        - Sadece p-değerine odaklanma
        - İş etkisini göz ardı etme
        - Sezonsal etkileri unutma
        """)


def show_theoretical_background():
    st.markdown('<p class="section-title">📖 Teorik Bilgiler</p>', unsafe_allow_html=True)

    with st.expander("🎯 A/B Testi Detaylı İnceleme", expanded=True):
        st.markdown("""
        ### 1. Test Öncesi Hazırlık

        #### A. Hipotez Oluşturma
        - **İş Problemi Tanımı**
          - Mevcut durum analizi
          - İyileştirme alanları
          - Beklenen etkiler

        - **Hipotez Formülasyonu**
          - H0 (Null Hipotez): Değişikliğin etkisi yok
          - H1 (Alternatif Hipotez): Değişiklik etkili

        #### B. Test Planı
        - **Örneklem Büyüklüğü Hesaplama**
          - Minimum detekte edilebilir etki
          - İstatistiksel güç
          - Güven seviyesi

        - **Test Süresi Belirleme**
          - İş döngüsü
          - Sezonsal etkiler
          - Yeterli veri toplama

        ### 2. Test Uygulama

        #### A. Veri Toplama
        - **Metrikler**
          - Birincil metrikler
          - İkincil metrikler
          - Kontrol metrikleri

        - **Segmentasyon**
          - Kullanıcı segmentleri
          - Davranış segmentleri
          - Teknik segmentler

        #### B. Kalite Kontrol
        - **Veri Doğrulama**
          - Eksik veri kontrolü
          - Aykırı değer analizi
          - Tutarlılık kontrolleri

        ### 3. İstatistiksel Analiz

        #### A. Varsayım Kontrolleri
        - **Normallik Testi**
          - Shapiro-Wilk testi
          - Q-Q plot analizi
          - Histogram inceleme

        - **Varyans Homojenliği**
          - Levene testi
          - Bartlett testi
          - F testi

        #### B. Test Seçimi
        - **Parametrik Testler**
          - Bağımsız örneklem t-testi
          - ANOVA
          - ANCOVA

        - **Non-parametrik Testler**
          - Mann-Whitney U testi
          - Kruskal-Wallis testi
          - Ki-kare testi
        """)


def run_analysis_with_explanation(control_df, test_df):
    st.markdown("""
    ## 📈 Analiz Süreci ve Detaylı İnceleme

    ### 1️⃣ Varsayım Kontrollerine Giriş
    İstatistiksel testleri seçmeden önce, verilerimizin hangi test için uygun olduğunu belirlemeliyiz. 
    Bu süreçte iki temel varsayımı kontrol edeceğiz:

    1. **Normallik Varsayımı**: Verilerimiz normal dağılıma uyuyor mu?
    2. **Varyans Homojenliği**: Grupların varyansları benzer mi?
    """)

    # Normallik Testi
    _, shapiro_pvalue_control = shapiro(control_df['Purchase'])
    _, shapiro_pvalue_test = shapiro(test_df['Purchase'])

    st.markdown("""
    ### 2️⃣ Normallik Testi (Shapiro-Wilk)

    Shapiro-Wilk testi, verilerin normal dağılıma uyup uymadığını kontrol eder.

    **Hipotezler:**
    - H0: Veriler normal dağılıma uyar
    - H1: Veriler normal dağılıma uymaz

    **Karar Kriteri:**
    - p > 0.05 ise H0 reddedilemez (normallik varsayımı sağlanır)
    - p < 0.05 ise H0 reddedilir (normallik varsayımı sağlanmaz)

    **Test Sonuçları:**
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Kontrol Grubu:**
        - p-değeri = {shapiro_pvalue_control:.4f}
        - Sonuç: {'Normal dağılıma uyuyor ✅' if shapiro_pvalue_control > 0.05 else 'Normal dağılıma uymuyor ❌'}
        """)

    with col2:
        st.info(f"""
        **Test Grubu:**
        - p-değeri = {shapiro_pvalue_test:.4f}
        - Sonuç: {'Normal dağılıma uyuyor ✅' if shapiro_pvalue_test > 0.05 else 'Normal dağılıma uymuyor ❌'}
        """)

    # Varyans Homojenliği
    _, levene_pvalue = levene(control_df['Purchase'], test_df['Purchase'])

    st.markdown("""
    ### 3️⃣ Varyans Homojenliği Testi (Levene)

    Levene testi, grupların varyanslarının homojen olup olmadığını kontrol eder.

    **Hipotezler:**
    - H0: Varyanslar homojendir
    - H1: Varyanslar homojen değildir

    **Karar Kriteri:**
    - p > 0.05 ise H0 reddedilemez (varyanslar homojen)
    - p < 0.05 ise H0 reddedilir (varyanslar homojen değil)
    """)

    st.info(f"""
    **Test Sonucu:**
    - p-değeri = {levene_pvalue:.4f}
    - Sonuç: {'Varyanslar homojen ✅' if levene_pvalue > 0.05 else 'Varyanslar homojen değil ❌'}
    """)

    # Test Seçimi ve Uygulama
    st.markdown("### 4️⃣ Uygun Test Seçimi")

    if shapiro_pvalue_control > 0.05 and shapiro_pvalue_test > 0.05:
        st.success("""
        **✅ Parametrik Test Kullanılacak**

        **Seçim Nedeni:**
        1. Normallik varsayımı her iki grup için de sağlandı
        2. Varyans homojenliği kontrol edildi

        **Seçilen Test:** Bağımsız İki Örneklem t-Testi
        """)

        test_stat, p_value = ttest_ind(control_df['Purchase'],
                                       test_df['Purchase'],
                                       equal_var=(levene_pvalue > 0.05))
        test_type = "Bağımsız İki Örneklem t-Testi"
    else:
        st.warning("""
        **⚠️ Non-parametrik Test Kullanılacak**

        **Seçim Nedeni:**
        1. Normallik varsayımı en az bir grup için sağlanmadı
        2. Bu durumda parametrik test kullanılamaz

        **Seçilen Test:** Mann-Whitney U Testi
        """)

        test_stat, p_value = mannwhitneyu(control_df['Purchase'],
                                          test_df['Purchase'])
        test_type = "Mann-Whitney U Testi"

    # Sonuçların Yorumlanması
    mean_diff = control_df['Purchase'].mean() - test_df['Purchase'].mean()

    st.markdown("### 5️⃣ Test Sonuçları ve Yorumlama")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Test İstatistiği",
            value=f"{test_stat:.4f}"
        )
    with col2:
        st.metric(
            label="p-değeri",
            value=f"{p_value:.4f}",
            delta="Anlamlı ❌" if p_value > 0.05 else "Anlamlı ✅"
        )

    st.markdown(f"""
    ### 6️⃣ Detaylı Analiz Sonuçları

    **1. Test Bilgileri:**
    - Kullanılan Test: {test_type}
    - Test İstatistiği: {test_stat:.4f}
    - p-değeri: {p_value:.4f}

    **2. Grup Karşılaştırması:**
    - Kontrol Grubu Ortalaması: {control_df['Purchase'].mean():.2f}
    - Test Grubu Ortalaması: {test_df['Purchase'].mean():.2f}
    - Ortalama Fark: {abs(mean_diff):.2f} birim ({('Kontrol grubu lehine' if mean_diff > 0 else 'Test grubu lehine')})

    **3. İstatistiksel Yorum:**
    {
    f"p-değeri ({p_value:.4f}) > 0.05 olduğundan H0 hipotezi reddedilemez. "
    f"Bu, iki grup arasında istatistiksel olarak anlamlı bir fark olmadığını gösterir. "
    if p_value > 0.05 else
    f"p-değeri ({p_value:.4f}) < 0.05 olduğundan H0 hipotezi reddedilir. "
    f"Bu, iki grup arasında istatistiksel olarak anlamlı bir fark olduğunu gösterir."
    }

    **4. Pratik Anlamlılık:**
    - Fark Büyüklüğü: %{abs(mean_diff / control_df['Purchase'].mean() * 100):.2f}
    - {
    'Bu fark istatistiksel olarak anlamlı değildir ve tesadüfi olabilir.'
    if p_value > 0.05 else
    'Bu fark istatistiksel olarak anlamlıdır ve gerçek bir etkiyi gösterir.'
    }

    **5. İş Etkisi:**
    {
    '- Mevcut sistemi değiştirmek için yeterli kanıt bulunmamaktadır.\n'
    '- Daha büyük örneklemle veya farklı değişikliklerle yeni testler planlanabilir.'
    if p_value > 0.05 else
    f'- {("Kontrol" if mean_diff > 0 else "Test")} grubu daha iyi performans göstermiştir.\n'
    '- Değişikliğin uygulanması önerilebilir.'
    }
    """)

    return {
        'shapiro_control': shapiro_pvalue_control,
        'shapiro_test': shapiro_pvalue_test,
        'levene': levene_pvalue,
        'test_type': test_type,
        'p_value': p_value,
        'mean_diff': mean_diff
    }


def create_visualizations(control_df, test_df):
    # Box Plot
    df = pd.concat([
        control_df.assign(group='Control'),
        test_df.assign(group='Test')
    ])

    fig_box = px.box(df, x='group', y='Purchase',
                     title='Satın Alma Dağılımı (Box Plot)',
                     color='group',
                     color_discrete_sequence=[MAIN_COLOR, SUCCESS_COLOR])

    # Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=control_df['Purchase'],
                                    name='Control',
                                    opacity=0.75,
                                    marker_color=MAIN_COLOR))
    fig_hist.add_trace(go.Histogram(x=test_df['Purchase'],
                                    name='Test',
                                    opacity=0.75,
                                    marker_color=SUCCESS_COLOR))
    fig_hist.update_layout(
        title='Satın Alma Dağılımı (Histogram)',
        xaxis_title='Satın Alma',
        yaxis_title='Frekans',
        barmode='overlay'
    )

    return fig_box, fig_hist


def main():
    st.sidebar.title("📚 İçerik Menüsü")
    page = st.sidebar.radio("Sayfa Seçin:",
                            ["🏠 Ana Sayfa",
                             "📖 Teorik Bilgiler",
                             "📊 Veri Analizi"])

    if page == "🏠 Ana Sayfa":
        show_welcome()

    elif page == "📖 Teorik Bilgiler":
        show_theoretical_background()

    elif page == "📊 Veri Analizi":
        st.markdown('<p class="section-title">📊 A/B Test Analizi</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Excel dosyanızı yükleyin (.xlsx)", type=['xlsx'])

        if uploaded_file:
            control_df, test_df = load_data(uploaded_file)

            if control_df is not None and test_df is not None:
                # Temel İstatistikler
                st.markdown('<p class="section-title">📈 Temel İstatistikler</p>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Kontrol Grubu (A)**")
                    st.dataframe(control_df.describe().round(2))

                with col2:
                    st.markdown("**Test Grubu (B)**")
                    st.dataframe(test_df.describe().round(2))

                # Görselleştirmeler
                st.markdown('<p class="section-title">📊 Veri Görselleştirme</p>', unsafe_allow_html=True)
                fig_box, fig_hist = create_visualizations(control_df, test_df)

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_box, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Hipotez Testi ve Detaylı Analiz
                results = run_analysis_with_explanation(control_df, test_df)


if __name__ == "__main__":
    main()

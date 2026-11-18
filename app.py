import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================================================================
# KONFIGURASI HALAMAN UTAMA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Pemantauan Stunting & Tumbuh Kembang",
    page_icon="📊",
    layout="wide"
)

# ==============================================================================
# FUNGSI PRAPEMROSESAN DATA STANDAR e-PPGBM
# ==============================================================================
def parse_usia_ke_bulan(usia_str):
    """Mengonversi teks 'X Tahun Y Bulan Z Hari' menjadi total angka bulan."""
    if pd.isna(usia_str):
        return np.nan
    tahun = re.search(r'(\d+)\s*Tahun', str(usia_str))
    bulan = re.search(r'(\d+)\s*Bulan', str(usia_str))
    t = int(tahun.group(1)) if tahun else 0
    b = int(bulan.group(1)) if bulan else 0
    return (t * 12) + b

@st.cache_data
def load_and_clean_data(file_source):
    df = pd.read_csv(file_source, dtype={'NIK': str, 'No HP': str})
    
    # Konversi format tanggal & usia
    df['Tanggal Pengukuran'] = pd.to_datetime(df['Tanggal Pengukuran'])
    df = df.sort_values(['NIK', 'Tanggal Pengukuran']).reset_index(drop=True)
    df['Bulan Ukur'] = df['Tanggal Pengukuran'].dt.to_period('M').astype(str)
    df['Usia_Bulan'] = df['Usia Saat Ukur'].apply(parse_usia_ke_bulan)
    
    # Deteksi 2T (Tidak Naik Berat Badan 2 Bulan Berturut-turut)
    df['Prev_NTOB'] = df.groupby('NIK')['Naik Berat Badan'].shift(1)
    df['Status_2T'] = ((df['Naik Berat Badan'] == 'T') & (df['Prev_NTOB'] == 'T'))
    
    # Encoding fitur untuk Machine Learning
    df['Is_Stunting'] = df['TB/U'].isin(['Pendek', 'Sangat Pendek']).astype(int)
    df['ASI_Num'] = (df['ASI Eksklusif'] == 'Ya').astype(int)
    df['JK_Num'] = (df['JK'] == 'L').astype(int)
    df['Tidak_Naik_BB'] = (df['Naik Berat Badan'] == 'T').astype(int)
    return df

# ==============================================================================
# SIDEBAR: UPLOAD DATA & NAVIGASI 4 HALAMAN
# ==============================================================================
st.sidebar.title("🧭 Navigasi Dashboard")
uploaded_file = st.sidebar.file_uploader("Unggah CSV Ekspor e-PPGBM (Opsional)", type=["csv"])

try:
    data_path = uploaded_file if uploaded_file is not None else "data_eppgbm_dummy.csv"
    df = load_and_clean_data(data_path)
except FileNotFoundError:
    st.error("File 'data_eppgbm_dummy.csv' tidak ditemukan. Pastikan berada dalam satu folder dengan app.py.")
    st.stop()

menu = st.sidebar.radio(
    "Pilih Menu Halaman:",
    [
        "1. Dashboard Kader & Desa",
        "2. Monitoring N/T/O/B & Alert 2T",
        "3. Prediksi Risiko Stunting (ML)",
        "4. Cek Pertumbuhan (Orang Tua)"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Sistem Pemantauan Gizi Balita Terintegrasi Format e-PPGBM")

# ==============================================================================
# HALAMAN 1: DASHBOARD ANALITIK KADER POSYANDU & DESA
# ==============================================================================
if menu == "1. Dashboard Kader & Desa":
    st.title("📊 Dashboard Pemantauan Gizi & Stunting Wilayah")
    st.markdown("Visualisasi agregat berbasis data **e-PPGBM** untuk membantu kader Posyandu dan perangkat desa menentukan prioritas intervensi.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        pilih_posyandu = st.multiselect(
            "Filter Posyandu:",
            options=sorted(df['Posyandu'].unique()),
            default=sorted(df['Posyandu'].unique())
        )
    with col_f2:
        pilih_bulan = st.selectbox(
            "Pilih Periode Pengukuran:",
            options=sorted(df['Bulan Ukur'].unique(), reverse=True)
        )

    df_filtered = df[(df['Posyandu'].isin(pilih_posyandu)) & (df['Bulan Ukur'] == pilih_bulan)]

    # Kartu Metrik Utama (KPI)
    total_anak = len(df_filtered)
    jml_stunting = int(df_filtered['Is_Stunting'].sum())
    prevalensi = (jml_stunting / total_anak * 100) if total_anak > 0 else 0
    jml_2t = int(df_filtered['Status_2T'].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Balita Diukur", f"{total_anak} Anak")
    m2.metric("Kasus Stunting (Pendek/Sangat Pendek)", f"{jml_stunting} Anak")
    m3.metric("Prevalensi Stunting", f"{prevalensi:.1f}%")
    m4.metric("Alarm Dini Kasus 2T", f"{jml_2t} Anak", delta="Perlu Rujukan", delta_color="inverse")

    st.divider()

    # Visualisasi Grafik
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Proporsi Status Tinggi Badan (TB/U)")
        fig_pie = px.pie(
            df_filtered, names='TB/U', hole=0.4, color='TB/U',
            color_discrete_map={
                'Normal': '#2ecc71',
                'Tinggi': '#3498db',
                'Pendek': '#f39c12',
                'Sangat Pendek': '#e74c3c'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Sebaran Status Gizi per Posyandu")
        rekap_pos = df_filtered.groupby(['Posyandu', 'TB/U']).size().reset_index(name='Jumlah')
        fig_bar = px.bar(
            rekap_pos, x='Posyandu', y='Jumlah', color='TB/U', barmode='group',
            color_discrete_map={
                'Normal': '#2ecc71',
                'Tinggi': '#3498db',
                'Pendek': '#f39c12',
                'Sangat Pendek': '#e74c3c'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tabel Prioritas & Tombol Download Laporan
    st.subheader("Daftar Balita Prioritas Intervensi (Stunting)")
    prioritas = df_filtered[df_filtered['Is_Stunting'] == 1][
        ['NIK', 'Nama Balita', 'JK', 'Usia Saat Ukur', 'Posyandu', 'Berat', 'Tinggi', 'ZS TB/U', 'TB/U', 'Naik Berat Badan']
    ].sort_values('ZS TB/U')
    
    st.dataframe(prioritas, use_container_width=True, hide_index=True)
    
    csv_prioritas = prioritas.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Daftar Prioritas Stunting (CSV)",
        data=csv_prioritas,
        file_name=f"prioritas_stunting_{pilih_bulan}.csv",
        mime="text/csv"
    )

# ==============================================================================
# HALAMAN 2: MONITORING KENAIKAN BERAT BADAN (N/T/O/B) & DETEKSI 2T
# ==============================================================================
elif menu == "2. Monitoring N/T/O/B & Alert 2T":
    st.title("⚖️ Monitoring Kenaikan Berat Badan (N/T/O/B) & Deteksi 2T")
    st.markdown("""
    * **N (Naik):** Kenaikan berat badan sesuai standar minimal (KBM).
    * **T (Tidak Naik):** Berat badan tetap, turun, atau naik kurang dari standar.
    * **O (Absen):** Bulan lalu tidak ditimbang.
    * **B (Baru):** Baru pertama kali ditimbang di Posyandu.
    """)

    rekap_ntob = df.groupby(['Bulan Ukur', 'Naik Berat Badan']).size().reset_index(name='Jumlah')
    fig_ntob = px.bar(
        rekap_ntob, x='Bulan Ukur', y='Jumlah', color='Naik Berat Badan', barmode='group',
        title="Tren Status N/T/O/B Setiap Bulan",
        color_discrete_map={'N': '#2ecc71', 'T': '#e74c3c', 'O': '#95a5a6', 'B': '#3498db'}
    )
    st.plotly_chart(fig_ntob, use_container_width=True)

    st.subheader("🚨 Daftar Balita Mengalami 2T (Tidak Naik 2 Bulan Berturut-turut)")
    kasus_2t = df[df['Status_2T'] == True][
        ['Bulan Ukur', 'NIK', 'Nama Balita', 'Posyandu', 'Usia Saat Ukur', 'Berat', 'TB/U', 'No HP']
    ]
    if len(kasus_2t) > 0:
        st.warning(f"Ditemukan **{len(kasus_2t)}** catatan balita mengalami 2T yang wajib dirujuk ke Puskesmas/TPG.")
        st.dataframe(kasus_2t, use_container_width=True, hide_index=True)
    else:
        st.success("Tidak ditemukan kasus 2T pada periode data ini.")

# ==============================================================================
# HALAMAN 3: PREDIKSI RISIKO STUNTING (MACHINE LEARNING)
# ==============================================================================
elif menu == "3. Prediksi Risiko Stunting (ML)":
    st.title("🤖 Analisis Faktor Risiko & Prediksi Stunting (Random Forest)")
    st.markdown("Pemodelan **Random Forest Classifier** untuk menganalisis faktor risiko dominan dan simulasi deteksi dini.")

    fitur = ['Usia_Bulan', 'JK_Num', 'BB Lahir', 'TB Lahir', 'Berat', 'Tinggi', 'LiLA', 'ASI_Num', 'Tidak_Naik_BB']
    X = df[fitur]
    y = df['Is_Stunting']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    st.info(f"Akurasi Model Random Forest pada Data Uji: **{acc * 100:.1f}%**")

    col_ml1, col_ml2 = st.columns([1.2, 1])

    with col_ml1:
        st.subheader("Faktor Paling Berpengaruh (Feature Importance)")
        imp_df = pd.DataFrame({
            'Variabel': ['Usia (Bulan)', 'Jenis Kelamin', 'Berat Lahir', 'Panjang Lahir', 'Berat Sekarang', 'Tinggi Sekarang', 'LiLA', 'ASI Eksklusif', 'Status Tidak Naik (T)'],
            'Tingkat Pengaruh': model.feature_importances_
        }).sort_values('Tingkat Pengaruh', ascending=True)

        fig_imp = px.bar(
            imp_df, x='Tingkat Pengaruh', y='Variabel', orientation='h',
            color='Tingkat Pengaruh', color_continuous_scale='Teal'
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_ml2:
        st.subheader("Simulasi Deteksi Dini Balita")
        with st.form("form_prediksi"):
            in_usia = st.slider("Usia Balita (Bulan)", 1, 60, 18)
            in_jk = st.selectbox("Jenis Kelamin", ["L", "P"])
            in_bbl = st.number_input("Berat Lahir (kg)", 1.5, 5.0, 2.9, step=0.1)
            in_tbl = st.number_input("Panjang Lahir (cm)", 40.0, 56.0, 48.5, step=0.5)
            in_bb = st.number_input("Berat Badan Saat Ini (kg)", 2.0, 25.0, 9.5, step=0.1)
            in_tb = st.number_input("Tinggi Badan Saat Ini (cm)", 45.0, 120.0, 76.0, step=0.5)
            in_lila = st.number_input("LiLA (cm)", 8.0, 22.0, 14.0, step=0.1)
            in_asi = st.selectbox("Riwayat ASI Eksklusif", ["Ya", "Tidak"])
            in_ntob = st.selectbox("Status Kenaikan BB Bulan Ini", ["N (Naik)", "T (Tidak Naik)"])
            submit = st.form_submit_button("Prediksi Risiko")

        if submit:
            input_data = pd.DataFrame([[
                in_usia,
                1 if in_jk == "L" else 0,
                in_bbl,
                in_tbl,
                in_bb,
                in_tb,
                in_lila,
                1 if in_asi == "Ya" else 0,
                1 if in_ntob.startswith("T") else 0
            ]], columns=fitur)
            prob = float(model.predict_proba(input_data)[0][1])
            if prob >= 0.5:
                st.error(f"⚠️ **Berisiko Tinggi Stunting** (Probabilitas: {prob*100:.1f}%). Segera konsultasikan ke petugas gizi Puskesmas.")
            else:
                st.success(f"✅ **Pertumbuhan Cenderung Normal** (Probabilitas Risiko: {prob*100:.1f}%). Pertahankan gizi seimbang.")

# ==============================================================================
# HALAMAN 4: PORTAL CEK PERTUMBUHAN MANDIRI UNTUK ORANG TUA
# ==============================================================================
elif menu == "4. Cek Pertumbuhan (Orang Tua)":
    st.title("👶 Buku KIA Digital: Cek Status Gizi Anak")
    st.markdown("Orang tua dapat memilih nama atau NIK anak untuk memantau riwayat dan grafik tumbuh kembang bulanan.")

    daftar_anak = df[['NIK', 'Nama Balita', 'Nama Ortu']].drop_duplicates().reset_index(drop=True)
    daftar_anak['Label'] = daftar_anak['Nama Balita'] + " (NIK: " + daftar_anak['NIK'] + " - Ortu: " + daftar_anak['Nama Ortu'] + ")"

    pilihan = st.selectbox("Cari Nama Anak / NIK:", options=daftar_anak['Label'].tolist())
    nik_terpilih = daftar_anak.loc[daftar_anak['Label'] == pilihan, 'NIK'].iloc[0]

    riwayat_anak = df[df['NIK'] == nik_terpilih].sort_values('Tanggal Pengukuran')
    terbaru = riwayat_anak.iloc[-1]

    st.subheader(f"Ringkasan Kondisi: {terbaru['Nama Balita']}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Usia Terakhir", terbaru['Usia Saat Ukur'])
    k2.metric("Berat Badan", f"{terbaru['Berat']} kg", f"{terbaru['BB/U']} ({terbaru['Naik Berat Badan']})")
    k3.metric("Tinggi Badan", f"{terbaru['Tinggi']} cm", terbaru['TB/U'])
    k4.metric("Status Gizi (BB/TB)", terbaru['BB/TB'])

    if terbaru['TB/U'] in ['Pendek', 'Sangat Pendek'] or terbaru['Status_2T']:
        st.warning("💡 **Catatan untuk Orang Tua:** Anak memerlukan perhatian khusus pada asupan protein hewani (telur, ikan, ayam) dan pemantauan rutin petugas gizi Posyandu.")
    else:
        st.success("🌟 **Catatan untuk Orang Tua:** Pertumbuhan anak terpantau baik. Teruskan pemantauan rutin setiap bulan di Posyandu.")

    g1, g2 = st.columns(2)
    with g1:
        fig_bb = px.line(riwayat_anak, x='Bulan Ukur', y='Berat', markers=True, title="Grafik Kenaikan Berat Badan (kg)")
        st.plotly_chart(fig_bb, use_container_width=True)

    with g2:
        fig_tb = px.line(riwayat_anak, x='Bulan Ukur', y='Tinggi', markers=True, title="Grafik Pertambahan Tinggi Badan (cm)")
        st.plotly_chart(fig_tb, use_container_width=True)

    st.subheader("Riwayat Pengukuran Posyandu")
    st.dataframe(
        riwayat_anak[['Tanggal Pengukuran', 'Posyandu', 'Usia Saat Ukur', 'Berat', 'Tinggi', 'Naik Berat Badan', 'ZS TB/U', 'TB/U']],
        use_container_width=True, hide_index=True
    )

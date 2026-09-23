"""Generate SVG and PNG diagram of the 1D CNN - BiLSTM + Attention architecture."""

import subprocess
from pathlib import Path

SVG_CONTENT = '''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1050" viewBox="0 0 1800 1050" role="img" aria-labelledby="title desc">
  <title id="title">Arsitektur Model Deteksi Jatuh 1D CNN - BiLSTM + Attention</title>
  <desc id="desc">Diagram detail arsitektur model deep learning deteksi jatuh HAFIBot: Ekstraksi Fitur, 1D CNN, BiLSTM, Attention Mechanism, Fully Connected Classifier, dan Posture Gating.</desc>

  <defs>
    <marker id="arrow" markerWidth="6" markerHeight="6" refX="5.5" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0 0 L6 3 L0 6 Z" fill="#000000"/>
    </marker>
    <style>
      @font-face { font-family: "Quicksand"; src: local("Quicksand"); }
      text { font-family: "Quicksand", "Segoe UI", sans-serif; fill: #000000; }
      .system { fill: #FFFFFF; stroke: #000000; stroke-width: 3; rx: 8; ry: 8; }
      .tab { fill: #7893C8; stroke: #000000; stroke-width: 3; rx: 6; ry: 6; }
      .block { fill: #FFFFFF; stroke: #000000; stroke-width: 2.5; rx: 6; ry: 6; }
      .block-highlight { fill: #FFF9E6; stroke: #000000; stroke-width: 2.5; rx: 6; ry: 6; }
      .block-attn { fill: #F0F4FC; stroke: #000000; stroke-width: 2.5; rx: 6; ry: 6; }
      .block-gate { fill: #EBF8F2; stroke: #000000; stroke-width: 2.5; rx: 6; ry: 6; }
      .branch { fill: #DCE5F6; stroke: #000000; stroke-width: 2.5; rx: 5; ry: 5; }
      .solid { fill: none; stroke: #000000; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; marker-end: url(#arrow); }
      
      .title-main { font-size: 24px; font-weight: 700; }
      .tab-title { font-size: 18px; font-weight: 700; fill: #FFFFFF; }
      .branch-title { font-size: 15px; font-weight: 700; }
      .label-head { font-size: 15.5px; font-weight: 700; }
      .label-body { font-size: 14px; font-weight: 600; }
      .caption { font-size: 12.5px; font-weight: 500; fill: #333333; }
      .dim-badge { fill: #000000; rx: 4; ry: 4; }
      .dim-text { font-size: 12px; font-weight: 700; fill: #FFFFFF; font-family: "Consolas", monospace; }
      .badge-blue { fill: #2B4C7E; rx: 4; ry: 4; }
      .badge-blue-text { font-size: 11.5px; font-weight: 700; fill: #FFFFFF; font-family: "Consolas", monospace; }
      .tag { fill: #E2E8F0; stroke: #000000; stroke-width: 1.5; rx: 4; ry: 4; }
      .tag-text { font-size: 11.5px; font-weight: 600; fill: #000000; }
    </style>
  </defs>

  <!-- Background Canvas -->
  <rect width="1800" height="1050" fill="#FFFFFF"/>

  <!-- Top Title Banner -->
  <rect x="40" y="20" width="1720" height="60" class="block" fill="#F8FAFC" stroke-width="3"/>
  <text x="900" y="58" class="title-main" text-anchor="middle">ARSITEKTUR MODEL DETEKSI JATUH: 1D CNN – BIDIRECTIONAL LSTM + ATTENTION</text>

  <!-- ============================================================ -->
  <!-- COLUMN 1: INPUT & REKAYASA FITUR (x: 40, width: 320) -->
  <!-- ============================================================ -->
  <rect x="40" y="115" width="320" height="900" class="system"/>
  <rect x="85" y="90" width="230" height="50" class="tab"/>
  <text x="200" y="122" class="tab-title" text-anchor="middle">1. Input &amp; Ekstraksi</text>

  <!-- Block 1.1: Video Source -->
  <rect x="65" y="160" width="270" height="85" class="block"/>
  <text x="200" y="188" class="label-head" text-anchor="middle">Input Aliran Video</text>
  <text x="200" y="210" class="caption" text-anchor="middle">Kamera Raspberry Pi NoIR</text>
  <text x="200" y="230" class="caption" text-anchor="middle">Resolusi: 640 × 480 @ 15 FPS</text>
  <path d="M200 245 V275" class="solid"/>

  <!-- Block 1.2: MediaPipe Pose -->
  <rect x="65" y="280" width="270" height="95" class="block"/>
  <text x="200" y="306" class="label-head" text-anchor="middle">MediaPipe Pose Estimator</text>
  <text x="200" y="328" class="caption" text-anchor="middle">33 Titik Landmark Kerangka</text>
  <rect x="95" y="340" width="210" height="24" class="tag"/>
  <text x="200" y="356" class="tag-text" text-anchor="middle">(x, y, z, visibility)</text>
  <path d="M200 375 V405" class="solid"/>

  <!-- Block 1.3: Landmark Selection -->
  <rect x="65" y="410" width="270" height="155" class="block"/>
  <text x="200" y="434" class="label-head" text-anchor="middle">Seleksi 13 Landmark Kunci</text>
  <text x="80" y="458" class="caption">• Kepala: Hidung (0)</text>
  <text x="80" y="478" class="caption">• Tubuh Atas: Bahu (11, 12), Siku (13, 14)</text>
  <text x="80" y="498" class="caption">• Lengan: Pergelangan Tangan (15, 16)</text>
  <text x="80" y="518" class="caption">• Pusat Massa: Pinggul (23, 24)</text>
  <text x="80" y="538" class="caption">• Kaki: Lutut (25, 26), Ankle (27, 28)</text>
  <path d="M200 565 V595" class="solid"/>

  <!-- Block 1.4: Temporal Feature Engineering -->
  <rect x="65" y="600" width="270" height="175" class="block-highlight"/>
  <text x="200" y="626" class="label-head" text-anchor="middle">Rekayasa Fitur Kinematik</text>
  <text x="80" y="650" class="caption">1. Koordinat Dasar (13 × 4) = 52 fitur</text>
  <text x="80" y="672" class="caption">2. Kecepatan / Velocity (13 × 3) = 39 fitur</text>
  <text x="80" y="694" class="caption">3. Percepatan / Accel (13 × 3) = 39 fitur</text>
  
  <rect x="80" y="710" width="240" height="26" class="badge-blue"/>
  <text x="200" y="728" class="badge-blue-text" text-anchor="middle">Total Fitur per Frame = 130</text>
  <text x="200" y="756" class="caption" text-anchor="middle">ΔPos = Pos(t) - Pos(t-1), ΔVel = Δ(ΔPos)</text>
  <path d="M200 775 V805" class="solid"/>

  <!-- Block 1.5: Sliding Window Buffer -->
  <rect x="65" y="810" width="270" height="110" class="block"/>
  <text x="200" y="836" class="label-head" text-anchor="middle">Sliding Window Buffer</text>
  <text x="200" y="858" class="caption" text-anchor="middle">Panjang Sekuens = 30 Frame (~2.0 s)</text>
  <text x="200" y="878" class="caption" text-anchor="middle">StandardScaler (Mean &amp; Std dev)</text>
  <rect x="85" y="888" width="230" height="24" class="dim-badge"/>
  <text x="200" y="904" class="dim-text" text-anchor="middle">Tensor Input: (Batch, 30, 130)</text>

  <!-- Connect Column 1 to Column 2 -->
  <path d="M335 865 H395" class="solid"/>

  <!-- ============================================================ -->
  <!-- COLUMN 2: 1D CNN FEATURE EXTRACTOR (x: 395, width: 320) -->
  <!-- ============================================================ -->
  <rect x="395" y="115" width="320" height="900" class="system"/>
  <rect x="440" y="90" width="230" height="50" class="tab"/>
  <text x="555" y="122" class="tab-title" text-anchor="middle">2. 1D CNN Spasial</text>

  <!-- Block 2.1: Permute 1 -->
  <rect x="420" y="160" width="270" height="75" class="block"/>
  <text x="555" y="188" class="label-head" text-anchor="middle">Permute / Transpose</text>
  <text x="555" y="210" class="caption" text-anchor="middle">Format Channels-First PyTorch</text>
  <rect x="440" y="218" width="230" height="22" class="dim-badge"/>
  <text x="555" y="233" class="dim-text" text-anchor="middle">Shape: (Batch, 130, 30)</text>
  <path d="M555 240 V270" class="solid"/>

  <!-- Block 2.2: Conv1D Block 1 -->
  <rect x="420" y="275" width="270" height="235" class="block"/>
  <rect x="435" y="290" width="240" height="32" class="branch"/>
  <text x="555" y="312" class="branch-title" text-anchor="middle">Conv1D Block #1 (64 Filter)</text>
  
  <rect x="440" y="335" width="230" height="30" class="tag"/>
  <text x="555" y="355" class="tag-text" text-anchor="middle">Conv1d(in=130, out=64, k=3, pad='same')</text>
  
  <rect x="440" y="375" width="230" height="26" class="tag"/>
  <text x="555" y="392" class="tag-text" text-anchor="middle">Batch Normalization (BatchNorm1d)</text>
  
  <rect x="440" y="410" width="230" height="26" class="tag"/>
  <text x="555" y="427" class="tag-text" text-anchor="middle">Aktivasi Non-Linear (ReLU)</text>
  
  <rect x="440" y="445" width="230" height="26" class="tag"/>
  <text x="555" y="462" class="tag-text" text-anchor="middle">Spatial Dropout (p = 0.3)</text>
  
  <rect x="440" y="480" width="230" height="22" class="badge-blue"/>
  <text x="555" y="495" class="badge-blue-text" text-anchor="middle">Output #1: (Batch, 64, 30)</text>
  <path d="M555 510 V540" class="solid"/>

  <!-- Block 2.3: Conv1D Block 2 -->
  <rect x="420" y="545" width="270" height="235" class="block"/>
  <rect x="435" y="560" width="240" height="32" class="branch"/>
  <text x="555" y="582" class="branch-title" text-anchor="middle">Conv1D Block #2 (128 Filter)</text>
  
  <rect x="440" y="605" width="230" height="30" class="tag"/>
  <text x="555" y="625" class="tag-text" text-anchor="middle">Conv1d(in=64, out=128, k=3, pad='same')</text>
  
  <rect x="440" y="645" width="230" height="26" class="tag"/>
  <text x="555" y="662" class="tag-text" text-anchor="middle">Batch Normalization (BatchNorm1d)</text>
  
  <rect x="440" y="680" width="230" height="26" class="tag"/>
  <text x="555" y="697" class="tag-text" text-anchor="middle">Aktivasi Non-Linear (ReLU)</text>
  
  <rect x="440" y="715" width="230" height="26" class="tag"/>
  <text x="555" y="732" class="tag-text" text-anchor="middle">Spatial Dropout (p = 0.3)</text>
  
  <rect x="440" y="750" width="230" height="22" class="badge-blue"/>
  <text x="555" y="765" class="badge-blue-text" text-anchor="middle">Output #2: (Batch, 128, 30)</text>
  <path d="M555 780 V810" class="solid"/>

  <!-- Block 2.4: Permute 2 (Back to Sequence) -->
  <rect x="420" y="815" width="270" height="100" class="block"/>
  <text x="555" y="842" class="label-head" text-anchor="middle">Permute / Transpose</text>
  <text x="555" y="864" class="caption" text-anchor="middle">Kembalikan ke Sequence First</text>
  <text x="555" y="882" class="caption" text-anchor="middle">Untuk Masukan Layer Rekuren</text>
  <rect x="440" y="890" width="230" height="22" class="dim-badge"/>
  <text x="555" y="905" class="dim-text" text-anchor="middle">Feature Map: (Batch, 30, 128)</text>

  <!-- Connect Column 2 to Column 3 -->
  <path d="M690 865 H750" class="solid"/>

  <!-- ============================================================ -->
  <!-- COLUMN 3: BIDIRECTIONAL LSTM (x: 750, width: 320) -->
  <!-- ============================================================ -->
  <rect x="750" y="115" width="320" height="900" class="system"/>
  <rect x="795" y="90" width="230" height="50" class="tab"/>
  <text x="910" y="122" class="tab-title" text-anchor="middle">3. Bi-LSTM Temporal</text>

  <!-- Block 3.1: Architecture Overview -->
  <rect x="775" y="160" width="270" height="100" class="block"/>
  <text x="910" y="188" class="label-head" text-anchor="middle">2-Layer Bi-LSTM</text>
  <text x="910" y="210" class="caption" text-anchor="middle">Hidden Size = 128 per Arah</text>
  <text x="910" y="228" class="caption" text-anchor="middle">Dropout Antar Layer = 0.3</text>
  <rect x="795" y="235" width="230" height="22" class="badge-blue"/>
  <text x="910" y="250" class="badge-blue-text" text-anchor="middle">Output Dim = 128 × 2 = 256</text>
  <path d="M910 260 V290" class="solid"/>

  <!-- Block 3.2: Forward & Backward Branches -->
  <rect x="775" y="295" width="270" height="370" class="block-attn"/>
  <text x="910" y="322" class="label-head" text-anchor="middle">Mekanisme Waktu Ganda</text>
  
  <!-- Forward Sub-block -->
  <rect x="790" y="340" width="240" height="105" class="block"/>
  <text x="910" y="364" class="branch-title" text-anchor="middle" fill="#067554">Forward LSTM (h_t_fwd)</text>
  <text x="910" y="386" class="caption" text-anchor="middle">Menganalisis Rangkaian Maju</text>
  <text x="910" y="406" class="caption" text-anchor="middle">t = 1 → t = 30 (Awal ke Akhir)</text>
  <rect x="805" y="415" width="210" height="22" class="tag"/>
  <text x="910" y="430" class="tag-text" text-anchor="middle">Ukuran Vektor: 128 Dimensi</text>

  <!-- Backward Sub-block -->
  <rect x="790" y="460" width="240" height="105" class="block"/>
  <text x="910" y="484" class="branch-title" text-anchor="middle" fill="#A11D1D">Backward LSTM (h_t_bwd)</text>
  <text x="910" y="506" class="caption" text-anchor="middle">Menganalisis Rangkaian Mundur</text>
  <text x="910" y="526" class="caption" text-anchor="middle">t = 30 → t = 1 (Konteks Balik)</text>
  <rect x="805" y="535" width="210" height="22" class="tag"/>
  <text x="910" y="550" class="tag-text" text-anchor="middle">Ukuran Vektor: 128 Dimensi</text>

  <!-- Concat Sub-block -->
  <rect x="790" y="580" width="240" height="70" class="block-highlight"/>
  <text x="910" y="604" class="label-head" text-anchor="middle">Concatenation Layer</text>
  <text x="910" y="626" class="caption" text-anchor="middle">h_t = [h_t_fwd ; h_t_bwd]</text>
  <text x="910" y="642" class="caption" text-anchor="middle">Gabungan Pola Maju &amp; Mundur</text>
  <path d="M910 665 V700" class="solid"/>

  <!-- Block 3.3: LSTM Hidden Sequence Output -->
  <rect x="775" y="705" width="270" height="210" class="block"/>
  <text x="910" y="732" class="label-head" text-anchor="middle">Sekuens Hidden State (H)</text>
  <text x="910" y="754" class="caption" text-anchor="middle">Representasi Temporal Utuh</text>
  <text x="910" y="774" class="caption" text-anchor="middle">Menyimpan Informasi Gerak</text>
  <text x="910" y="794" class="caption" text-anchor="middle">dari Seluruh 30 Timestep</text>
  
  <rect x="795" y="810" width="230" height="24" class="dim-badge"/>
  <text x="910" y="826" class="dim-text" text-anchor="middle">H: (Batch, 30, 256)</text>
  
  <text x="910" y="860" class="caption" text-anchor="middle">H = [h_1, h_2, ..., h_30]</text>
  <text x="910" y="880" class="caption" text-anchor="middle">Setiap h_t berukuran 256 unit</text>

  <!-- Connect Column 3 to Column 4 -->
  <path d="M1045 810 H1105" class="solid"/>

  <!-- ============================================================ -->
  <!-- COLUMN 4: ATTENTION MECHANISM (x: 1105, width: 320) -->
  <!-- ============================================================ -->
  <rect x="1105" y="115" width="320" height="900" class="system"/>
  <rect x="1145" y="90" width="240" height="50" class="tab"/>
  <text x="1265" y="122" class="tab-title" text-anchor="middle">4. Bahdanau Attention</text>

  <!-- Block 4.1: Attention Scoring -->
  <rect x="1130" y="160" width="270" height="245" class="block-attn"/>
  <text x="1265" y="188" class="label-head" text-anchor="middle">Attention Scoring Network</text>
  <text x="1265" y="210" class="caption" text-anchor="middle">Additive (Bahdanau-Style)</text>
  
  <rect x="1145" y="225" width="240" height="28" class="tag"/>
  <text x="1265" y="244" class="tag-text" text-anchor="middle">Linear Layer: (256 → 256)</text>
  
  <rect x="1145" y="260" width="240" height="26" class="tag"/>
  <text x="1265" y="278" class="tag-text" text-anchor="middle">Aktivasi Non-Linear: Tanh()</text>
  
  <rect x="1145" y="293" width="240" height="28" class="tag"/>
  <text x="1265" y="312" class="tag-text" text-anchor="middle">Linear Layer: (256 → 1)</text>
  
  <rect x="1145" y="328" width="240" height="26" class="tag"/>
  <text x="1265" y="346" class="tag-text" text-anchor="middle">Softmax Normalization (dim=1)</text>
  
  <rect x="1145" y="365" width="240" height="24" class="badge-blue"/>
  <text x="1265" y="381" class="badge-blue-text" text-anchor="middle">Bobot α_t: (Batch, 30, 1)</text>
  <path d="M1265 405 V440" class="solid"/>

  <!-- Block 4.2: Adaptive Frame Weighting Concept -->
  <rect x="1130" y="445" width="270" height="210" class="block-highlight"/>
  <text x="1265" y="472" class="label-head" text-anchor="middle">Pembobotan Adaptif</text>
  <text x="1265" y="494" class="caption" text-anchor="middle">Memberikan Bobot Signifikan (α)</text>
  <text x="1265" y="514" class="caption" text-anchor="middle">pada Frame Benturan Jatuh</text>
  
  <!-- Mini Bar Chart of Attention -->
  <g transform="translate(1160, 530)">
    <rect x="0" y="35" width="12" height="15" fill="#A0AEC0"/>
    <rect x="18" y="38" width="12" height="12" fill="#A0AEC0"/>
    <rect x="36" y="32" width="12" height="18" fill="#A0AEC0"/>
    <rect x="54" y="25" width="12" height="25" fill="#A0AEC0"/>
    <rect x="72" y="18" width="12" height="32" fill="#3182CE"/>
    <rect x="90" y="5" width="12" height="45" fill="#E53E3E"/>
    <rect x="108" y="0" width="12" height="50" fill="#E53E3E"/>
    <rect x="126" y="8" width="12" height="42" fill="#E53E3E"/>
    <rect x="144" y="22" width="12" height="28" fill="#3182CE"/>
    <rect x="162" y="30" width="12" height="20" fill="#A0AEC0"/>
    <rect x="180" y="38" width="12" height="12" fill="#A0AEC0"/>
    <rect x="198" y="42" width="12" height="8" fill="#A0AEC0"/>
  </g>
  <text x="1265" y="605" class="tag-text" text-anchor="middle">Fase Benturan (Impact Frame)</text>
  <text x="1265" y="625" class="caption" text-anchor="middle">Mereduksi Noise Frame Berdiri / Statis</text>
  <text x="1265" y="643" class="caption" text-anchor="middle">Σ α_t = 1.0 (t = 1...30)</text>
  <path d="M1265 655 V690" class="solid"/>

  <!-- Block 4.3: Weighted Sum Context Vector -->
  <rect x="1130" y="695" width="270" height="220" class="block"/>
  <text x="1265" y="722" class="label-head" text-anchor="middle">Weighted Sum Context</text>
  <text x="1265" y="744" class="caption" text-anchor="middle">c = Σ (α_t × h_t)  [t = 1...30]</text>
  <text x="1265" y="768" class="caption" text-anchor="middle">Menggabungkan Fitur Temporal</text>
  <text x="1265" y="788" class="caption" text-anchor="middle">Berdasarkan Relevansi Kejadian</text>
  
  <rect x="1145" y="805" width="240" height="26" class="dim-badge"/>
  <text x="1265" y="822" class="dim-text" text-anchor="middle">Context Vector c: (Batch, 256)</text>
  
  <text x="1265" y="856" class="caption" text-anchor="middle">Vektor Tunggal 256 Dimensi</text>
  <text x="1265" y="876" class="caption" text-anchor="middle">Siap Masuk ke Classifier</text>

  <!-- Connect Column 4 to Column 5 -->
  <path d="M1400 818 H1460" class="solid"/>

  <!-- ============================================================ -->
  <!-- COLUMN 5: CLASSIFIER & POSTURE GATING (x: 1460, width: 300) -->
  <!-- ============================================================ -->
  <rect x="1460" y="115" width="300" height="900" class="system"/>
  <rect x="1495" y="90" width="230" height="50" class="tab"/>
  <text x="1610" y="122" class="tab-title" text-anchor="middle">5. Klasifikasi &amp; Gate</text>

  <!-- Block 5.1: Fully Connected Layers -->
  <rect x="1475" y="160" width="270" height="240" class="block"/>
  <rect x="1490" y="175" width="240" height="30" class="branch"/>
  <text x="1610" y="196" class="branch-title" text-anchor="middle">Dense Classification</text>
  
  <rect x="1490" y="215" width="240" height="28" class="tag"/>
  <text x="1610" y="234" class="tag-text" text-anchor="middle">Linear Layer: (256 → 64)</text>
  
  <rect x="1490" y="250" width="240" height="26" class="tag"/>
  <text x="1610" y="267" class="tag-text" text-anchor="middle">Aktivasi ReLU()</text>
  
  <rect x="1490" y="283" width="240" height="26" class="tag"/>
  <text x="1610" y="300" class="tag-text" text-anchor="middle">Dropout (p = 0.4)</text>
  
  <rect x="1490" y="316" width="240" height="28" class="tag"/>
  <text x="1610" y="335" class="tag-text" text-anchor="middle">Linear Layer: (64 → 2 Kelas)</text>
  
  <rect x="1490" y="352" width="240" height="26" class="tag"/>
  <text x="1610" y="369" class="tag-text" text-anchor="middle">Softmax Output: P(ADL), P(Fall)</text>

  <path d="M1610 400 V435" class="solid"/>

  <!-- Block 5.2: Decision Threshold -->
  <rect x="1475" y="440" width="270" height="90" class="block"/>
  <text x="1610" y="466" class="label-head" text-anchor="middle">Threshold Evaluasi</text>
  <rect x="1495" y="478" width="230" height="26" class="badge-blue"/>
  <text x="1610" y="495" class="badge-blue-text" text-anchor="middle">Threshold Jatuh: P(Fall) ≥ 0.80</text>
  <text x="1610" y="520" class="caption" text-anchor="middle">Teruji Optimal (Fall Recall 100%)</text>
  <path d="M1610 530 V565" class="solid"/>

  <!-- Block 5.3: Posture Gating & Geometric Filter -->
  <rect x="1475" y="570" width="270" height="195" class="block-gate"/>
  <rect x="1490" y="585" width="240" height="30" class="branch" fill="#C3EADB"/>
  <text x="1610" y="606" class="branch-title" text-anchor="middle">Posture Gating Verification</text>
  
  <text x="1490" y="632" class="caption">• Torso Angle: θ ≥ 50° (Tegak)</text>
  <text x="1490" y="652" class="caption">• Aspect Ratio: H/W ≥ 1.25</text>
  <text x="1490" y="672" class="caption">• Vertical Disp: dy_torso ≥ 0.04</text>
  
  <rect x="1490" y="685" width="240" height="35" class="tag" fill="#FFFFFF"/>
  <text x="1610" y="700" class="tag-text" text-anchor="middle">Filter False Positive:</text>
  <text x="1610" y="714" class="caption" text-anchor="middle">Blokir alarm jika tubuh terbukti tegak</text>
  
  <text x="1610" y="748" class="caption" text-anchor="middle">(Duduk, Sujud, Membungkuk)</text>
  <path d="M1610 765 V800" class="solid"/>

  <!-- Block 5.4: Final Decision Output -->
  <rect x="1475" y="805" width="270" height="185" class="block"/>
  <text x="1610" y="830" class="label-head" text-anchor="middle">Keputusan &amp; Tindakan</text>
  
  <!-- ADL Result Box -->
  <rect x="1490" y="842" width="115" height="50" class="block" fill="#F0FFF4" stroke="#276749"/>
  <text x="1547" y="865" class="label-head" text-anchor="middle" fill="#276749">ADL</text>
  <text x="1547" y="882" class="caption" text-anchor="middle" fill="#276749">Normal</text>

  <!-- Fall Result Box -->
  <rect x="1615" y="842" width="115" height="50" class="block" fill="#FFF5F5" stroke="#9B2C2C"/>
  <text x="1672" y="865" class="label-head" text-anchor="middle" fill="#9B2C2C">FALL</text>
  <text x="1672" y="882" class="caption" text-anchor="middle" fill="#9B2C2C">Darurat</text>

  <!-- Robot & IoT Action -->
  <rect x="1490" y="902" width="240" height="75" class="tag" fill="#F7FAFC"/>
  <text x="1610" y="920" class="tag-text" text-anchor="middle">Respons Otomatis Saat Fall:</text>
  <text x="1610" y="938" class="caption" text-anchor="middle">1. Latch E-Stop Motor Robot</text>
  <text x="1610" y="955" class="caption" text-anchor="middle">2. Buzzer Alarm Jetson &amp; Chatbot</text>
  <text x="1610" y="972" class="caption" text-anchor="middle">3. Web Push Notification ke PWA</text>

</svg>'''

def main():
    out_svg = Path("docs/hafibot_model_architecture.svg")
    out_png = Path("docs/hafibot_model_architecture.png")
    
    out_svg.write_text(SVG_CONTENT, encoding="utf-8")
    print(f"SVG written to {out_svg}")
    
    edge_path = Path(r"C:Program Files (x86)MicrosoftEdgeApplicationmsedge.exe")
    if edge_path.exists():
        abs_svg = out_svg.resolve().as_uri()
        abs_png = str(out_png.resolve()).replace("\\", "/")
        cmd = [
            str(edge_path),
            "--headless=old",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            f"--screenshot={abs_png}",
            "--window-size=1800,1050",
            abs_svg,
        ]
        print("Rendering PNG via Edge...")
        subprocess.run(cmd, check=True)
        print(f"PNG rendered to {out_png}")
    else:
        print("Edge not found; SVG ready for direct viewing.")

if __name__ == "__main__":
    main()

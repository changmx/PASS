"""Offline, typeset formula references for the independent GUI tools."""
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QVBoxLayout

from PASS.gui.tool_math import FormulaBrowser

MASS_REFERENCE_FORMULAS = r"""
<h2>质量数据与来源</h2>
<p>运行时读取离线文件 <b>PASS/tool/mass_catalog.json</b>。文件由官方整表下载后程序转换生成，
不是逐个网页抄录。每条记录包含可读名称、原表行号；元数据保存下载网址、版本和 SHA-256。
A 为整数核子数，Z 为质子数，q 为电荷态。</p>
<ul>
<li><a href="https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt">IAEA AMDC · AME2020</a>：3558 条基态核素记录，其中 1008 条有估算标记；主要为中性原子质量，含自由中子记录。</li>
<li><a href="https://physics.nist.gov/cuu/Constants/Table/allascii.txt">NIST · CODATA 2022</a>：u 换算、电子、缪子、质子、中子和轻核质量。</li>
<li><a href="https://pdg.lbl.gov/2026/mcdata/mass_width_2026.txt">Particle Data Group · PDG 2026</a>：319 个有数值质量的编号；文件含共振态，GUI 开放其中的常用粒子。</li>
<li><a href="https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html">NIST ASD · Ionization Energies</a>：6019 条电离阶段记录，其中 172 条能量未知；电荷态质量按需计算，并未逐个存成独立质量表。</li>
<li><a href="https://webbook.nist.gov/cgi/cbook.cgi?ID=C12385136&amp;Mask=20">NIST WebBook · Hydrogen</a>：H− 电子亲和能，Lykke 等 (1991)。</li>
</ul>
<p>不是所有可能的同位素或电荷态都能计算。缺少核素质量或任何必要电离能时明确报错。
不含核同质异能态质量；负离子仅支持 H−。ASD 电离能未计同位素位移，不能把所有结果视为同等精度的直接测量。</p>
<h3>先计算当前离子的完整静止质量</h3>
<p>M<sub>atom</sub> 是原子质量的 u 数值，I<sub>j</sub> 是从 j+ 到 (j+1)+ 的电离能。正离子：</p>
<eq>E_0=m_0c^2=M_{\mathrm{atom}}\,uc^2-q\,m_ec^2+\sum_{j=0}^{q-1}I_j</eq>
<p>电离能项为正。质子、氘核、氚核、氦-3 核和 α 粒子优先用 CODATA 直接值。
H− 使用原子质量加电子质量再减电子亲和能。表内不确定度及估算标记保留；未合成包含所有相关性的总不确定度。</p>
<h2>实际质量比 μ 与能量归一化</h2>
<p>μ 无量纲，数值等于该粒子的完整静止质量以 u 表示时的数值；它不是整数质量数 A。</p>
<p>同一核素改变电荷态，质子数 Z、中子数 N 与质量数 A = Z + N 均不变。
电子数及束缚能改变，因此实际静止质量和 μ 会改变。相邻正电荷态的基态关系为：</p>
<eq>m_{q+1}c^2=m_qc^2-m_ec^2+I_q</eq>
<eq>\mu=\frac{m_0}{u}=\frac{E_0}{uc^2},\qquad uc^2=931.49410372\ \mathrm{MeV}</eq>
<p>离子/原子按整数核子数 A 归一化：Ek = K/A，单位 AMeV，即 MeV/核子。
电子、缪子、τ 轻子和介子没有核子数（A=0），采用单粒子动能 MeV。
中子及反核子 A=1，每核子量与单粒子量相同。统一记除数 D=max(A,1)：</p>
<eq>D=\max(A,1),\qquad E_k=\frac{K}{D},\qquad E_r=\frac{E}{D},\qquad p_r=\frac{p}{D}</eq>
<eq>K=D E_k,\qquad E=D E_r,\qquad p=D p_r</eq>
<eq>m_r=\frac{m_0}{D},\qquad \gamma=1+\frac{D E_k}{m_0c^2}</eq>
<p>界面先显示静止质量 m<sub>0</sub>/A（MeV/c²），再显示完整粒子的 μ；A=0 时显示完整静止质量 m<sub>0</sub>。
μ 仅表示实际质量比，不作为能量除数。平均每核子质量包含当前电荷态的电子和束缚能修正，不能直接用 u 替代。
固定 A 和 Ek 改变 q 时，完整动能 K 不变，但 m<sub>0</sub>、μ、γ、β、动量和磁刚度会随之变化。
数据出处与覆盖范围保存在 mass_catalog.json 的 metadata 中。</p>
"""

BEAM_FORMULAS = MASS_REFERENCE_FORMULAS + r"""
<h2>运动学与磁刚度</h2>
<p>下列 K、E、p 为完整粒子量；显示的归一化值须乘 D 得到完整量。</p>
<eq>E=E_0+K,\qquad \gamma=1+\frac{K}{E_0}</eq>
<eq>\beta=\sqrt{1-\frac{1}{\gamma^2}},\qquad v=\beta c</eq>
<eq>pc=\sqrt{K(K+2E_0)},\qquad \beta\gamma=\frac{pc}{E_0}</eq>
<eq>B\rho=\frac{p}{|q|e}</eq>
<p>使用显示的 p<sub>r</sub>（离子为 p/A，A=0 为 p，单位 GeV/c）时，磁刚度数值（T·m）为：</p>
<eq>B\rho=\frac{D\,p_r}{0.299792458\,|q|}</eq>
<h3>反向换算</h3>
<eq>K=\frac{(pc)^2}{\sqrt{(pc)^2+E_0^2}+E_0}</eq>
<p>上式等价于总能量减静止能量，避免低能时两个相近数直接相减。</p>
<eq>\gamma=\frac{1}{\sqrt{1-\beta^2}},\qquad K=(\gamma-1)E_0</eq>
<p>要求动能非负、总能量不低于静止能量、γ≥1、0≤β&lt;1。
零动能时 β=0、γ=1、动量为零。中性粒子没有磁刚度，不能用电流反算束流功率。</p>
<h2>电流与束流动能功率</h2>
<p>I 是电流大小，粒子率为 Ṅ，K<sub>J</sub> 是单粒子动能（J）。</p>
<eq>I=\dot{N}|q|e,\qquad P=\dot{N}K_J</eq>
<eq>P=\frac{I K_J}{|q|e},\qquad I=\frac{P|q|e}{K_J}</eq>
<p>输入 I（mA）、E<sub>k</sub>（离子为 AMeV，A=0 为 MeV），输出 P（kW）时的数值换算：</p>
<eq>P=\frac{I\,D E_k}{|q|}</eq>
<p>不需要环周长。电流和功率须采用同一时间口径；动能功率不含静止能量，也不等于电网输入功率。
动能和功率均为零时不能唯一反算电流，零动能不能对应非零动能功率。</p>
<h3>瞬时值与峰值</h3>
<eq>P(t)=\frac{I(t)K_J(t)}{|q|e}</eq>
<p>适用于同一截面、同一时刻；有能散时采用粒子流加权的平均动能。
只有动能固定，电流峰值才对应功率峰值。仅有平均值不能确定峰值。</p>
<h2>脉冲束流</h2>
<p>N<sub>pulse</sub> 为每脉冲真实粒子数，f<sub>rep</sub> 为实际脉冲/引出重复频率，τ 为脉宽。</p>
<eq>I_{\mathrm{avg}}=N_{\mathrm{pulse}}|q|e f_{\mathrm{rep}}</eq>
<eq>W_{\mathrm{pulse}}=N_{\mathrm{pulse}}K_J,\qquad P_{\mathrm{avg}}=W_{\mathrm{pulse}} f_{\mathrm{rep}}</eq>
<eq>I_{\mathrm{pulse}}=\frac{N_{\mathrm{pulse}}|q|e}{\tau},\qquad P_{\mathrm{pulse}}=\frac{W_{\mathrm{pulse}}}{\tau}</eq>
<p>脉冲内值为脉宽内平均值，平顶脉冲时才等于峰值。不提供脉宽时只计算长期平均值和每脉冲能量。</p>
<eq>\tau>0,\qquad 0\leq\tau f_{\mathrm{rep}}\leq1</eq>
<h2>环内循环电流与储能</h2>
<p>C 为环周长，N<sub>ring</sub> 是所有束团的真实粒子总数，不是宏粒子数。</p>
<eq>f_{\mathrm{rev}}=\frac{\beta c}{C},\qquad T_{\mathrm{rev}}=\frac{C}{\beta c}</eq>
<eq>I_{\mathrm{ring}}=N_{\mathrm{ring}}|q|e f_{\mathrm{rev}},\qquad W_{\mathrm{stored}}=N_{\mathrm{ring}}K_J</eq>
<p>循环粒子不会每圈都被引出，不能把循环频率当作引出重复频率计算靶上功率。零速度时周期无有限值。</p>
<h3>常数</h3>
<eq>c=299792458\ \mathrm{m/s},\qquad e=1.602176634\times10^{-19}\ \mathrm{C}</eq>
<p>工具使用上述固定评估质量；追踪输入遵循其已有接口单位。</p>
"""

TUNE_FORMULAS = r"""
<h2>横向共振条件</h2>
<eq>mQ_x+nQ_y=l,\qquad r=|m|+|n|</eq>
<p>Q<sub>x</sub>、Q<sub>y</sub> 为完整 tune；m、n、l 为整数，m、n 不能同时为零。
r 是共振阶数。坐标范围可以跨整数或使用负值。</p>
<h3>共振类型</h3>
<p>m=0 或 n=0 是单平面共振，mn&gt;0 是和共振，mn&lt;0 是差共振。
差共振使用虚线，其余为实线。颜色区分阶数，工作点另用形状区分。</p>
<h3>增减、去重与显示</h3>
<p>自动线按 1–12 阶独立勾选。几何去重同时约分 (m,n,l)，并统一符号。</p>
<eq>2Q_x=2\ \Longleftrightarrow\ Q_x=1</eq>
<p>上面两条线重合，归入 1 阶；2Q<sub>x</sub>=1 仍为 2 阶。
关闭低阶后，重合线不会因开启高阶重新出现。自定义线可以添加、隐藏、删除，与自动线重合时按自定义样式高亮。
仅绘制窗口内的线段；工作点表格可以编辑名称、坐标、颜色和符号。
名称默认为空，可保持留空；非空名称显示在绘图区右上角的紧凑图例，颜色和符号与散点一致。
空名称只绘制散点；隐藏点与范围外点不进入图例。</p>
<h3>图的含义</h3>
<p>此处仅绘制共振几何位置，不计算驱动项、共振宽度或动态孔径。
点到线的远近不能单独判断束流稳定性；工作点不会修改追踪输入。</p>
"""


class FormulaDialog(QDialog):

    def __init__(self, title, html, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 740)
        layout = QVBoxLayout(self)
        self.browser = FormulaBrowser(html, self)
        self.browser.render()
        layout.addWidget(self.browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.copy_formulas = buttons.addButton("复制公式（LaTeX）", QDialogButtonBox.ActionRole)
        self.copy_formulas.clicked.connect(
            lambda: QApplication.clipboard().setText("\n\n".join(f"\\[{tex}\\]" for tex in self.browser.rendered_equations)))
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

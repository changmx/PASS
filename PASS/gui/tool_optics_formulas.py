"""Typeset RF, transverse optics, magnet and Exciter references."""
from PASS.gui.tool_formulas import MASS_REFERENCE_FORMULAS

RF_FORMULAS = MASS_REFERENCE_FORMULAS + r"""
<h2>单谐波 RF bucket</h2>
<p>采用固定参数、小动量偏差的光滑近似。本节采用上文的 D=max(A,1)，总能量 E<sub>r</sub>=E/D、归一化电荷量 q<sub>r</sub>=|q|/D。</p>
<eq>E_r=\frac{m_0c^2}{D}+E_k,\qquad q_r=\frac{|q|}{D}</eq>
<eq>\eta=\frac{1}{\gamma_t^2}-\frac{1}{\gamma^2},\qquad \phi=\phi_s-\frac{2\pi h z_{\mathrm{rel}}}{C}</eq>
<p>h 是 RF 谐波数，与 Injection 分组谐波独立。φ<sub>s</sub> 是已含腔相位、偏置和束团中心贡献的有效相位。</p>
<eq>\frac{d\phi}{dN}=2\pi h\eta\delta</eq>
<eq>\frac{d\delta}{dN}=\frac{q_r V}{\beta^2 E_r}\left(\sin\phi-\sin\phi_s\right)</eq>
<p>N 为圈数，δ=Δp/p，V 为正的电压幅值。电荷符号通过有效相位表示，与当前 PASS 约定一致。</p>
<eq>\eta\cos\phi_s<0</eq>
<p>上式为稳定条件。低于跃迁常用 0°，高于跃迁常用 180°。
电荷、V、动能或 η 为零，以及 90°/270° 退化相位均不支持稳定桶。</p>
<h3>哈密顿量与分离曲线</h3>
<eq>\theta=\phi-\phi_s,\qquad a=2\pi h\eta,\qquad b=\frac{q_r V}{\beta^2 E_r}</eq>
<eq>H=\frac{|a|\delta^2}{2}+U(\theta)</eq>
<eq>U(\theta)=\mathrm{sign}(\eta)b\left[\cos(\phi_s+\theta)-\cos\phi_s+\theta\sin\phi_s\right]</eq>
<p>同步点为 U=0 的稳定极小值，取相邻两侧不稳定点较低的势垒作为 H<sub>sep</sub>，数值求出转向点。</p>
<eq>\delta_{\pm}=\pm\sqrt{\frac{2[H_{\mathrm{sep}}-U(\theta)]}{|a|}}</eq>
<p>内部轨道使用介于 0 与 H<sub>sep</sub> 之间的能量水平，支持加速和减速相位。</p>
<h3>能量坐标、频率与桶面积</h3>
<eq>\Delta E_r\simeq\beta^2E_r\delta</eq>
<p>离子能量图轴和半高显示 ΔE/A（MeV），A=0 时显示 ΔE（MeV）。半高是最大正偏差，宽度报告全宽。</p>
<eq>f_{\mathrm{rev}}=\frac{\beta c}{C},\qquad f_{\mathrm{RF}}=h f_{\mathrm{rev}}</eq>
<eq>Q_s=\sqrt{\frac{-h\eta q_r V\cos\phi_s}{2\pi\beta^2E_r}},\qquad f_s=Q_s f_{\mathrm{rev}}</eq>
<eq>\Delta E_{r,s}=q_r V\sin\phi_s</eq>
<eq>\mathcal{A}_r=\frac{2\beta^2E_r}{2\pi f_{\mathrm{RF}}}\int_{\phi_L}^{\phi_R}\delta_+(\phi)\,d\phi</eq>
<p>桶面积为完整粒子桶面积除以 D，离子单位为 eV·s/核子，A=0 时为 eV·s。静止桶的动量半高及相位全宽为：</p>
<eq>\delta_{\max}=\sqrt{\frac{2q_r V}{\pi h|\eta|\beta^2 E_r}},\qquad \Delta\phi=2\pi</eq>
<p>不包含多谐波、集体效应、辐射、捕获 ramp。接近跃迁或较大 δ 时须使用更完整模型；不能替代逐圈追踪。</p>
"""

EMITTANCE_FORMULAS = MASS_REFERENCE_FORMULAS + r"""
<h2>RMS 发射度与 Twiss 参数</h2>
<p>Twiss β（m）、α、γ（1/m）描述横向相空间；相对论因子用下标 rel 区分。</p>
<eq>\gamma=\frac{1+\alpha^2}{\beta},\qquad \alpha=\pm\sqrt{\beta\gamma-1}</eq>
<p>β 必须为正，反算须满足 βγ≥1。α 的正负不能由 β、γ 唯一决定，需要选择分支。</p>
<eq>\varepsilon_n=\beta_{\mathrm{rel}}\gamma_{\mathrm{rel}}\varepsilon</eq>
<p>输入 1 π·mm·mrad 时，公式采用 ε=10⁻⁶ m·rad；π 是椭圆面积约定，不额外乘 π 或乘 4。
几何和归一化发射度均使用 RMS 定义。零动能时由归一化值反算几何值不唯一。</p>
<h3>协方差矩阵的分量与色散贡献</h3>
<p>假设 betatron 坐标与相对动量偏差 δ 不相关；其 RMS 为 σ<sub>δ</sub>，色散为 D、D′。</p>
<eq>\Sigma_{11}=\sigma_x^2=\beta\varepsilon+(D\sigma_\delta)^2</eq>
<eq>\Sigma_{22}=\sigma_{x'}^2=\gamma\varepsilon+(D'\sigma_\delta)^2</eq>
<eq>\Sigma_{12}=\Sigma_{21}=\mathrm{Cov}(x,x')=-\alpha\varepsilon+DD'\sigma_\delta^2</eq>
<eq>\varepsilon_{\mathrm{proj}}=\sqrt{\det\Sigma},\qquad r=\frac{\mathrm{Cov}(x,x')}{\sigma_x\sigma_{x'}}</eq>
<p>投影包含不同动量粒子的色散位移。Cov 衡量位置和角度的共同变化，r 为归一化相关系数。
D=D′=0 时，投影与 betatron 椭圆相同。</p>
<eq>\varepsilon=\frac{\sigma_x^2-(D\sigma_\delta)^2}{\beta}</eq>
  <p>由束斑反算时，σ<sub>x</sub> 不得小于 |D|σ<sub>δ</sub>。一个束斑测量不能同时确定未知 Twiss 和色散。</p>
  <h3>由投影 RMS 与相关性反算</h3>
  <p>输入 σx、σx′ 及 r 或 Cov；这些都是扣除质心后的投影统计量。r 模式要求两个 RMS 为正且 |r|≤1。
  协方差模式允许零 RMS，但此时协方差必须为零，相关系数未定义。</p>
  <eq>B_{11}=\sigma_x^2-D^2\sigma_\delta^2,\qquad B_{22}=\sigma_{x'}^2-D'^2\sigma_\delta^2</eq>
  <eq>B_{12}=\mathrm{Cov}(x,x')-DD'\sigma_\delta^2,\qquad \mathrm{Cov}(x,x')=r\sigma_x\sigma_{x'}</eq>
  <eq>\varepsilon=\sqrt{\det B},\quad \beta=\frac{B_{11}}{\varepsilon},\quad \alpha=-\frac{B_{12}}{\varepsilon},\quad \gamma=\frac{B_{22}}{\varepsilon}</eq>
  <p>投影矩阵和扣除色散后的 B 均须半正定；不相容输入报错。ε=0 时反算 Twiss 未定义，
  保留零发射度及退化为线段或点的曲线。输入 Twiss 的原有模式仍保留给定 Twiss。</p>
  <p>每页独立计算，参考粒子和能量共用；多页叠加不计算合束发射度。
  中心 x₀、x′₀ 同时平移 betatron 和投影曲线，不改变 RMS、协方差、发射度或 Twiss。</p>
  <h3>相椭圆及包含比例</h3>
<eq>\gamma x^2+2\alpha xx'+\beta x'^2=n^2\varepsilon</eq>
<eq>\mathcal{A}=\pi n^2\varepsilon,\qquad F(n)=1-\exp\left(-\frac{n^2}{2}\right)</eq>
<p>投影椭圆使用 ε<sub>proj</sub>。非退化二维高斯在 n=1 椭圆内约含 39.35%，不是 68%。
x′ 为傍轴角度；在参考动量附近，PASS 的归一化横向动量近似对应此角度。
不处理横向耦合、非高斯分布和 betatron–能散相关项。</p>
"""

MAGNET_FORMULAS = MASS_REFERENCE_FORMULAS + r"""
<h2>带符号磁刚度</h2>
<eq>B\rho=\frac{p}{qe}=\frac{D p_r}{qe}</eq>
<p>p 是完整粒子动量大小，q 带符号。因此电子的磁刚度为负；束流计算器显示的是磁刚度大小。
可以由参考粒子和动能计算，也可以直接输入带符号磁刚度（T·m）。</p>
<h3>二极铁</h3>
<eq>k_0=\frac{B}{B\rho}=\frac{1}{\rho},\qquad \theta=k_0L=K_{0L}</eq>
<eq>\int B\,dl=BL=(B\rho)\theta</eq>
<p>L 为有效弧长，不是弦长。角度输入为 deg、积分强度输出为 rad。
零磁场时半径无有限值，零半径不合法。</p>
<h3>四、六、八极与理想极面场</h3>
<p>默认输入归一化强度 k<sub>1</sub>、k<sub>2</sub>、k<sub>3</sub>。四、六、八极分别对应 n=1、2、3。</p>
<eq>G_n=\frac{\partial^n B_y}{\partial x^n},\qquad k_n=\frac{G_n}{B\rho},\qquad K_{nL}=k_n L</eq>
<eq>B_p[\mathrm{Gauss}]=10^4(B\rho)k_n\frac{r^n}{n!}</eq>
<p>半径 r 输入 mm，计算时转为 m；极面场保留所选极面的极性，场大小为 |B<sub>p</sub>|。</p>
<eq>B_{p,1}=G_1r,\qquad B_{p,2}=\frac{G_2r^2}{2},\qquad B_{p,3}=\frac{G_3r^3}{6}</eq>
<p>这一行场值采用 Tesla。六、八极的 2!、3! 与 PASS 多极展开一致，不能省略。</p>
<eq>\Delta p_x=-K_{nL}\frac{r^n}{n!}\quad (x=r,\ y=0)</eq>
<h3>四极薄透镜焦距</h3>
<eq>f_x\simeq\frac{1}{K_{1L}},\qquad f_y\simeq-\frac{1}{K_{1L}}</eq>
<eq>\int G_1\,dl=G_1L=(B\rho)K_{1L}</eq>
<p>零梯度对应无有限焦距。薄透镜估算不是有限长四极铁从端面测量的精确焦距。
场、梯度和 K 值均保留符号；改变参考电荷而保持物理磁场时，归一化强度符号改变。</p>
<h3>螺线管</h3>
<p>默认输入归一化 K<sub>s</sub>。κ 是傍轴 Larmor 聚焦参数：</p>
<eq>K_s=\frac{B_z}{B\rho},\qquad \kappa=\frac{K_s}{2},\qquad \theta=\kappa L</eq>
<eq>B_z[\mathrm{Gauss}]=10^4(B\rho)K_s,\qquad \int B_z\,dl=(B\rho)K_sL</eq>
<eq>\frac{1}{f}\simeq\kappa^2L</eq>
<p>焦距仅适用于弱薄透镜近似；有限长螺线管包含两平面耦合。正 θ 对应 PASS 的旋转：</p>
<eq>x_{\mathrm{rot}}=x\cos\theta+y\sin\theta,\qquad y_{\mathrm{rot}}=y\cos\theta-x\sin\theta</eq>
<p>显示的是轴向场；孔径半径和 K<sub>s</sub> 不足以确定铁芯极面场，需要磁路几何与场模型。
不包含边缘场、饱和、磁滞、多极误差或励磁电流标定。</p>
"""

EXCITER_FORMULAS = MASS_REFERENCE_FORMULAS + r"""
<h2>激励器基准踢角</h2>
<p>与 PASS Exciter 的单频 FM、单频 FM+AM、双频 FM、双频 FM+AM 对应。
V 为电压幅值，d 为极板间距，L 为极板有效长。</p>
<eq>A_0=\frac{VL}{d\,\beta c\,|B\rho|},\qquad E_{\mathrm{plate}}=\frac{V}{d}</eq>
<eq>t_{\mathrm{plate}}=\frac{L}{\beta c},\qquad f_0=\frac{\beta c}{C}</eq>
<p>双频叠加包络可达 2A<sub>0</sub>。负电荷沿用当前 PASS 的磁刚度大小约定，极性通过电极/信号相位解释。</p>
<eq>f_c=Q_{\mathrm{excite}}f_0,\qquad \Delta f=\Delta Q\,f_0</eq>
<p>Δf 是扫频全宽；激励 tune 可以包含整数边带，例如 9.47。这里只预览设定信号，不预测共振响应。</p>
<h3>到达时间</h3>
<p>t<sub>elapsed</sub> 从激励开始计时，t<sub>0,start</sub> 是启动时参考钟。粒子坐标不折叠：</p>
<eq>t_{\mathrm{arrive}}=t_{0,\mathrm{start}}+t_{\mathrm{elapsed}}-\frac{z_{\mathrm{rel}}+z_{\mathrm{center}}}{\beta c}</eq>
<eq>\tau=t_{\mathrm{arrive}}\ \mathrm{mod}\ T</eq>
<p>只有信号内部将到达时间对扫频周期 T 取余。</p>
<h3>单频 FM</h3>
<eq>\varphi=2\pi f_c\tau+\frac{\pi\Delta f}{T}\tau(\tau-T)</eq>
<eq>\mathrm{kick}=A_0\sin\varphi,\qquad f=f_c+\Delta f\left(\frac{\tau}{T}-\frac{1}{2}\right)</eq>
<h3>双频 FM</h3>
<p>前半周期（0≤τ≤T/2）和后半周期（T/2&lt;τ&lt;T）分别使用：</p>
<eq>\varphi_1=2\pi f_c\tau+\pi\Delta f\left(f_d\tau-\frac{1}{2}\right)\tau</eq>
<eq>\varphi_2=2\pi f_c\tau+\pi\Delta f\left(\tau-\frac{T}{2}\right)(f_d\tau-1)</eq>
<eq>\mathrm{kick}=2A_0\cos\left(\frac{\pi\Delta f\tau}{2}\right)\sin\varphi</eq>
<p>两支频率由分段相位导数及包络分解得到：</p>
<eq>f_{\pm,1}=f_c+\Delta f f_d\tau-\frac{\Delta f}{4}\pm\frac{\Delta f}{4}</eq>
<eq>f_{\pm,2}=f_c+\Delta f f_d\tau-\frac{\Delta f}{2}-\frac{\Delta f f_d T}{4}\pm\frac{\Delta f}{4}</eq>
<p>周期重置或半周期拼接可能产生宽频成分；这些分段频率不能视为严格带宽。</p>
<h3>AM 扩散模型</h3>
<p>AM 按有效圈数更新，定义以下中间量以便阅读：</p>
<eq>n=\lfloor t_{\mathrm{elapsed}}f_0\rfloor,\qquad t=\frac{n}{f_0}</eq>
<eq>a=\exp\left(-\frac{r_0^2}{\delta_0^2}\right),\qquad b(t)=a+\frac{t}{t_{\mathrm{ext}}}(1-a)</eq>
<eq>F_{\mathrm{AM}}=\sqrt{\frac{r_0^2(1-a)}{[\ln b(t)]^2\,[t_{\mathrm{ext}}a+t(1-a)]\,f_0 k}}</eq>
<eq>\mathrm{kick}_{\mathrm{FM+AM}}=F_{\mathrm{AM}}\,\mathrm{kick}_{\mathrm{FM}}</eq>
<p>r<sub>0</sub>、δ<sub>0</sub> 用同一长度单位，k 是原 Exciter 模型参数。
模型在 t<sub>ext</sub> 处发散，绘图区间必须在其之前。本工具不对扩散模型作额外物理标定。</p>
<h3>采样与导出</h3>
<p>均匀时间采样用于画波形；逐圈点单独按 n/f<sub>0</sub> 的到达相位计算。
FFT 使用 Hann 窗并按窗增益归一化，输出单边踢角振幅谱。</p>
<eq>\Delta f_{\mathrm{FFT}}=\frac{1}{T_{\mathrm{window}}}</eq>
<p>采样峰值是当前窗口数值峰值，并非任意时间的严格上界。最多 200000 个采样点，窗口过长时需要缩短。
CSV 使用明确的 SI 单位；不计算束流响应、损失、反馈、随机噪声或发射度增长。</p>
"""

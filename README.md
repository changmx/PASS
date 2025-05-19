# PASS



# 软件介绍





# 使用方法





# 物理模型

## 传输元件

### 坐标转换

在传输过程中为了方便起见使用$(x,p_x,y,p_y,\tau,p_t)$坐标来描述粒子的运动状态，因此需要对粒子原本坐标$(x,p_x,y,p_y,z,\frac{\Delta p}{p})$进行转换。$\tau$与$p_t$的定义与MADX中定义相同：

$\tau$是粒子与参考粒子之间的时间差乘以光速$c$，即：
$$
\tau = c\Delta t = c\frac{z}{v}=\frac{z}{\beta_0}
$$
$p_t$是粒子的能量偏差除以参考粒子动量与光速的乘积，即：
$$
p_t = \frac{\Delta E}{P_0c} = \frac{\Delta E}{mvc} = \frac{\Delta E}{mc^2\times \frac{v}{c}}
=\frac{\Delta E}{\beta_0 E_0}
\\
\frac{\Delta p}{p} \approx \frac{1}{\beta_0^2}\frac{\Delta E}{E_0}
\\
p_t = \beta_0 \frac{\Delta p}{p}
$$
这里动量分散与能量分散之间的关系采用了$1/\beta_0^2$的近似，在S.Y.Lee加速器物理第四版P225页有更精确公式，在MADX使用手册第39章也有更精确描述，这两种方法在能量较高时精度比$1/\beta_0^2$要高，但是在低能时（能量小于4MeV/u）误差很大，因此还是采用$1/\beta_0^2$公式。

### 概念介绍

粒子感受到的磁铁的力可以写作：
$$
\vec{F}=q\vec{v} \times \vec{B}=
q
\begin{pmatrix}
0\\0\\v_z
\end{pmatrix}
\times
\begin{pmatrix}
B_x\\
B_y\\
0
\end{pmatrix}
=
qv_z
\begin{pmatrix}
-B_y\\
B_x\\
0
\end{pmatrix}
$$
对于长度为$L$的多极磁铁，薄透镜近似模型下，磁铁的横向冲击为：
$$
p_x = \int F_x dt = -\int q v_z B_y dt = -qB_y\int v_zdt \approx -qB_yL
\\
p_y = \int F_y dt = \int q v_z B_x dt = qB_x\int v_zdt \approx qB_xL
$$
偏角为：
$$
\Delta \theta_x = \frac{p_x}{p_0}=\frac{-qB_yL}{qB\rho}=-\frac{B_yL}{B\rho}
\\
\Delta \theta_y = \frac{p_y}{p_0}=\frac{qB_xL}{qB\rho}=\frac{B_xL}{B\rho}
$$
多极场的表达式为：
$$
B_y+iB_x=\sum_{n=0}^{\infty}(b_n+ia_n)(x+iy)^n
\\
b_n=\frac{1}{n!}\frac{\partial ^n B_y}{\partial x^n}
,\quad
a_n=\frac{1}{n!}\frac{\partial ^n B_x}{\partial x^n}
$$
这里$b_n$，$a_n$表示$2(n+1)$极磁铁的系数，$b_n$对应于正磁铁元件，$a_n$对应于斜磁铁元件。

由麦克斯韦方程可得：
$$
\vec{\nabla}\cdot \vec{B}=0 \rightarrow
\frac{\partial B_x}{\partial x}+\frac{\partial B_y}{\partial y}=0
\\
\vec{\nabla} \times \vec{B}=0 \rightarrow
\frac{\partial B_y}{\partial x} = \frac{\partial B_x}{\partial y}
$$

### 漂移节

对于一个长度为$L$漂移节，我们可以把粒子在其中的传输写作：
$$
\vec{x}(s=L)= R\cdot \vec{x}(s=0)
$$
其中：
$$
\vec{x} = 
\begin{pmatrix}
x\\p_x\\y\\p_y\\\tau\\p_t
\end{pmatrix}
,\quad

R = 
\begin{pmatrix}
1&L&0&0&0&0\\
0&1&0&0&0&0\\
0&0&1&L&0&0\\
0&0&0&1&0&0\\
0&0&0&0&1&\frac{L}{\beta^2\gamma^2}\\
0&0&0&0&0&1
\end{pmatrix}
$$

### 二极磁铁

对于一个长度为$L$的二极磁铁，我们可以把粒子在其中的传输写作：
$$
\vec{x}(s=L)= R_{fr}\cdot R\cdot R_{fl}\cdot \vec{x}(s=0)
$$
其中$R_{fl}$与$R_{fr}$为二极磁铁入口与出口处边缘场的影响，$R$为二极磁铁本身的影响，定义如下变量：

- $\rho$，弯转半径
- $h=1/\rho$，即为曲率
- $e_1$，同MADX，二极铁入口边缘角
- $e_2$，同MADX，二极铁出口边缘角
- FINT，同MADX，边缘角积分
- FINTX，同MADX，边缘角积分
- HGAP，同MADX，磁铁间隙的一半
- $\psi_1 = e_1-2\times HGAP\times h \times FINT\times (1+\sin^2(e_1))/\cos(e_1)$
- $\psi_2 = e_2-2\times HGAP\times h \times FINTX\times (1+\sin^2(e_2))/\cos(e_2)$

$$
R_{fl} =
\begin{pmatrix}
1&0&0&0&0&0\\
\frac{h\tan(e_1)}{1+\Delta p/p} &1&0&0&0&0\\
0&0&1&0&0&0\\
0&0&-\frac{h\tan(\psi_1)}{1+\Delta p/p}&1&0&0\\
0&0&0&0&1&0\\
0&0&0&0&0&1
\end{pmatrix}
,
\quad

R_{fr} =
\begin{pmatrix}
1&0&0&0&0&0\\
\frac{h\tan(e_2)}{1+\Delta p/p} &1&0&0&0&0\\
0&0&1&0&0&0\\
0&0&-\frac{h\tan(\psi_2)}{1+\Delta p/p}&1&0&0\\
0&0&0&0&1&0\\
0&0&0&0&0&1
\end{pmatrix}
$$

$$
R = 
\begin{pmatrix}
\cos(hL)&\frac{sin(hL)}{h}&0&0&0&\frac{1-\cos(hL)}{h\beta_0}\\
-h\sin(hL)&\cos(hL)&0&0&0&\frac{\sin(hL)}{\beta_0}\\
0&0&1&L&0&0\\
0&0&0&1&0&0\\
-\frac{\sin(hL)}{\beta_0}&-\frac{1-\cos((hL))}{h\beta_0}&0&0&1&\frac{\sin(hL)}{h\beta_0^2}-L\\
0&0&0&0&0&1
\end{pmatrix}
$$

对于动量存在偏差的粒子，在边缘场传输过程中对聚焦强度进行了修正：$K_n(\delta)=\frac{K_n}{1+\delta}$。

### 厚正四极磁铁

对于一个长度为$L$的四极磁铁，我们可以把粒子在其中的传输写作：
$$
\vec{x}(s=L)= R_{f/d}\cdot \vec{x}(s=0)
$$
定义$\omega=\sqrt{\frac{|k_1|}{1+\delta}}$，这里对动量存在偏差的粒子，在传输过程中对聚焦强度进行了修正，$k_1>0$表示水平聚焦。聚焦磁铁与散焦磁铁的传输矩阵分别为$R_f$与$R_d$：
$$
R_f = 
\begin{pmatrix}
\cos{(\omega L)}&\frac{\sin({\omega L})}{\omega}&0&0&0&0\\
-\omega\sin{(\omega L)}&\cos{(\omega L)}&0&0&0&0\\
0&0&\cosh{(\omega L)}&\frac{\sinh{(\omega L)}}{\omega}&0&0\\
0&0&\omega\sinh{(\omega L)}&\cosh{(\omega L)}&0&0\\
0&0&0&0&1&\frac{L}{\beta_0^2\gamma_0^2}\\
0&0&0&0&0&1
\end{pmatrix}
$$

$$
R_d = 
\begin{pmatrix}
\cosh{(\omega L)}&\frac{\sinh{(\omega L)}}{\omega}&0&0&0&0\\
\omega\sinh{(\omega L)}&\cosh{(\omega L)}&0&0&0&0\\
0&0&\cos{(\omega L)}&\frac{\sin({\omega L})}{\omega}&0&0\\
0&0&-\omega\sin{(\omega L)}&\cos{(\omega L)}&0&0\\
0&0&0&0&1&\frac{L}{\beta_0^2\gamma_0^2}\\
0&0&0&0&0&1
\end{pmatrix}
$$

### 厚斜四极磁铁

对于一个长度为$L$的斜四极磁铁，当$k_{1s}>0$时，其对粒子运动状态的影响可以看作聚焦四极磁铁旋转+45度后的结果，当$k_{1s}<0$时，其效果既可以看作散焦四极铁旋转+45度的效果，也可以看作聚焦四极铁旋转-45度的效果，两者等价：
$$
\vec{x}(s=L)= R(-\frac{\pi}{2})\cdot R_{f/d} \cdot(\frac{\pi}{2})\cdot \vec{x}(s=0)
$$
定义$\omega=\sqrt{\frac{|k_{1s}|}{1+\delta}}$，这里对动量存在偏差的粒子，在传输过程中对聚焦强度进行了修正，$k_{1s}>0$表示散焦。散焦正四极磁铁与散焦正四极磁铁的传输矩阵分别为$R_f$与$R_d$，把粒子坐标$(x,y)$旋转$\theta$的矩阵为：
$$
R(\theta) = 
\begin{pmatrix}
\cos\theta&0&\sin\theta&0\\
0&\cos\theta&0&\sin\theta\\
-\sin\theta&0&\cos\theta&0\\
0&-\sin\theta&0&\cos\theta
\end{pmatrix}
$$
定义：

- $C_p = \frac{\cos(\omega L)+\cosh(\omega L)}{2}$
- $C_m = \frac{\cos(\omega L)-\cosh(\omega L)}{2}$
- $S_p = \frac{\sin(\omega L)+\sinh(\omega L)}{2}$
- $S_m = \frac{\sin(\omega L)-\sinh(\omega L)}{2}$

使用Mathematic计算得到：
$$
R(k_{1s}>0) = 
\begin{pmatrix}
C_p&\frac{S_p}{\omega}&C_m&\frac{S_m}{\omega}&0&0\\
-\omega S_m&C_p&-\omega S_p&C_m&0&0\\
C_m&\frac{S_m}{\omega}&C_p&\frac{S_p}{\omega}&0&0\\
-\omega S_p&C_m&-\omega S_m&C_p&0&0\\
0&0&0&0&1&\frac{L}{\beta_0^2\gamma_0^2}\\
0&0&0&0&0&1
\end{pmatrix}
$$

$$
R(k_{1s}<0)=
\begin{pmatrix}
C_p&\frac{S_p}{\omega}&-C_m&-\frac{S_m}{\omega}&0&0\\
-\omega S_m&C_p&\omega S_p&-C_m&0&0\\
-C_m&-\frac{S_m}{\omega}&C_p&\frac{S_p}{\omega}&0&0\\
\omega S_p&-C_m&-\omega S_m&C_p&0&0\\
0&0&0&0&1&\frac{L}{\beta_0^2\gamma_0^2}\\
0&0&0&0&0&1
\end{pmatrix}
$$

### 薄正四极磁铁

对于长度为$L$的四极磁铁，使用之前介绍的公式计算可得：
$$
b_1 =\frac{\partial B_y}{\partial x }
\\
B_y = b_1x=\frac{\partial B_y}{\partial x }x
\\
B_x = b_1y=\frac{\partial B_y}{\partial x }y
\\
K_1 = \frac{1}{B\rho} \frac{\partial B_y}{\partial x }
\\
\Delta \theta_x = -K_1Lx,\quad \Delta \theta_y = K_1Ly
$$

### 薄斜四极磁铁

对于长度为$L$的四极磁铁，使用之前介绍的公式计算可得：
$$
a_1 =\frac{\partial B_x}{\partial x }
\\
B_y = -a_1y=-\frac{\partial B_x}{\partial x }y
\\
B_x = a_1x=\frac{\partial B_x}{\partial x }x
\\
K_{1s} = \frac{1}{B\rho} \frac{\partial B_x}{\partial x } = 
\frac{1}{2B\rho} (\frac{\partial B_x}{\partial x}-\frac{\partial B_y}{\partial y})
\\
\Delta \theta_x = K_{1s}Ly, \quad \Delta \theta_y = K_{1s}Lx
$$

### 正六极磁铁

对于长度为$L$的六极磁铁，使用之前介绍的公式计算可得：
$$
B_y+iB_x=(b_2+ia_2)(x+iy)^2=(b_2+ia_2)(x^2+2ixy-y^2)
\\
b_2=\frac{1}{2}\frac{\partial ^2 B_y}{\partial x^2}
\\
B_y = b_2(x^2-y^2)=\frac{1}{2}\frac{\partial ^2 B_y}{\partial x^2}(x^2-y^2)
\\
B_x = b_1\times2xy=\frac{\partial ^2 B_y}{\partial x^2}xy
\\
K_2 = \frac{1}{B\rho} \frac{\partial B_y^2}{\partial x^2 }
\\
\Delta \theta_x = -\frac{1}{2}K_2L(x^2-y^2),\quad \Delta \theta_y = K_2Lxy
$$

### 斜六极磁铁

对于长度为$L$的六极磁铁，使用之前介绍的公式计算可得：
$$
B_y+iB_x=(b_2+ia_2)(x+iy)^2=(b_2+ia_2)(x^2+2ixy-y^2)
\\
a_2=\frac{1}{2}\frac{\partial ^2 B_x}{\partial x^2}
\\
B_y = -2a_2xy=-\frac{\partial ^2 B_y}{\partial x^2}xy
\\
B_x = a_2\times(x^2-y^2)=\frac{1}{2}\frac{\partial ^2 B_y}{\partial x^2}(x^2-y^2)
\\
K_{2s} = \frac{1}{B\rho} \frac{\partial B_x^2}{\partial x^2 }
\\
\Delta \theta_x = K_{2s}Lxy,\quad \Delta \theta_y = \frac{1}{2}K_{2s}L(x^2-y^2)
$$

### 八极磁铁

对于长度为$L$的六极磁铁，使用之前介绍的公式计算可得：
$$
B_y+iB_x=(b_3+ia_3)(x+iy)^3=(b_3+ia_3)(x^3 - 3 x y^2 + i (3 x^2 y - y^3))
\\
a_3 = \frac{1}{6}\frac{\partial ^3 B_x}{\partial x^3}, \quad b_3 = \frac{1}{6}\frac{\partial ^3 B_y}{\partial x^3}
$$
对于正八极磁铁来说：
$$
B_y = b_3(x^3-3xy^2)=\frac{1}{6}\frac{\partial ^3 B_y}{\partial x^3}(x^3-3xy^2)
\\
B_x = b_3(3x^2y-y^3)=\frac{1}{6}\frac{\partial ^3 B_y}{\partial x^3}(3x^2y-y^3)
\\
K_{3} = \frac{1}{B\rho}\frac{\partial ^3 B_y}{\partial x^3}
\\
\Delta \theta_x = -\frac{1}{6}K_{3}L(x^3-3xy^2), \quad \Delta \theta_y = \frac{1}{6}K_{3}L(3x^2y-y^3)
$$
对于斜八极磁铁来说：
$$
B_y = -a_3(3x^2y-y^3)=-\frac{1}{6}\frac{\partial ^3 B_x}{\partial x^3}(3x^2y-y^3)
\\
B_x = a_3(x^3-3xy^2)=\frac{1}{6}\frac{\partial ^3 B_y}{\partial x^3}(x^3-3xy^2)
\\
K_{3s} = \frac{1}{B\rho}\frac{\partial ^3 B_x}{\partial x^3}
\\
\Delta \theta_x = \frac{1}{6}K_{3s}L(3x^2y-y^3), \quad \Delta \theta_x = \frac{1}{6}K_{3s}L(x^3-3xy^2)
$$

### 多极磁铁

对于长度为$L$的多极磁铁，使用之前介绍的公式计算可得：
$$
K_n = \frac{1}{B\rho}\frac{\partial ^n B_y}{\partial x^n}, \quad K_{sn} = \frac{1}{B\rho}\frac{\partial ^n B_x}{\partial x^n}
$$

$$
(x+iy)^0=1
\\
(x+iy)^1=x+iy
\\
(x+iy)^2=x^2 + 2 i x y - y^2
\\
(x+iy)^3=x^3 - 3 x y^2 + i (3 x^2 y - y^3)
\\
(x+iy)^4=x^4 - 6 x^2 y^2 + y^4 + i (4 x^3 y - 4 x y^3)
\\
(x+iy)^5=x^5 - 10 x^3 y^2 + 5 x y^4 + i (5 x^4 y - 10 x^2 y^3 + y^5)
\\
(x+iy)^6=x^6 - 15 x^4 y^2 + 15 x^2 y^4 - y^6 + i (6 x^5 y - 20 x^3 y^3 + 6 x y^5)
\\
(x+iy)^7=x^7 - 21 x^5 y^2 + 35 x^3 y^4 - 7 x y^6 + i (7 x^6 y - 35 x^4 y^3 + 21 x^2 y^5 - y^7)
\\
(x+iy)^8=x^8 - 28 x^6 y^2 + 70 x^4 y^4 - 28 x^2 y^6 + y^8 +  i (8 x^7 y - 56 x^5 y^3 + 56 x^3 y^5 - 8 x y^7)
\\
(x+iy)^9=x^9 - 36 x^7 y^2 + 126 x^5 y^4 - 84 x^3 y^6 + 9 x y^8 +  i (9 x^8 y - 84 x^6 y^3 + 126 x^4 y^5 - 36 x^2 y^7 + y^9)
\\
(x+iy)^{10}=x^{10} - 45 x^8 y^2 + 210 x^6 y^4 - 210 x^4 y^6 + 45 x^2 y^8 - y^{10} + i (10 x^9 y - 120 x^7 y^3 + 252 x^5 y^5 - 120 x^3 y^7 + 10 x y^9)
$$

对于正多极铁来说：
$$
\Delta \theta_x = -K_nL\times \frac{1}{n!}\times Re\{ (x+iy)^n  \}
\\
\Delta \theta_x = K_nL\times \frac{1}{n!}\times Im\{ (x+iy)^n  \}
$$
对于斜多极铁来说：
$$
\Delta \theta_x = K_{sn}L\times \frac{1}{n!}\times Im\{ (x+iy)^n  \}
\\
\Delta \theta_x = K_{sn}L\times \frac{1}{n!}\times Re\{ (x+iy)^n  \}
$$

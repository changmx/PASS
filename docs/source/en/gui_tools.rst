GUI tools
=========

The **Tools** workspace contains the beam calculator, tune diagram,
**RF bucket绘制**, phase-space plotting and emittance calculation, magnet conversion, and
Exciter preview. Its left section buttons share the configuration library's
style and have individual icons. Values persist while switching pages in the
current window. Tools do not modify the active tracking input or saved project.
Each page provides a separate **Detailed formulas** window and result copying.
Formulas are typeset offline with fractions, radicals, sums and real subscripts;
no web engine or online math assets are required. **Copy formulas (LaTeX)** retains
access to their editable mathematical source.

Particles and authoritative masses
-----------------------------------

Enter ion **A**, signed charge state **q**, and proton number **Z**, or choose a
named particle. Non-ion choices include electrons/positrons, muons/antimuons,
taus, charged/neutral pions and kaons, neutrons/antineutrons, and antiprotons.
A/q/Z are disabled for these named species; A and Z alone cannot distinguish
an electron, muon or pion. Neutral massive particles support kinematics but
have no magnetic rigidity or current-to-power conversion.

Search accepts Chinese/English names, symbols, isotope notation, and approximate
English matches, for example ``carbon``, ``C-12``, ``238U35+``, ``muon``, ``pi+``.
Selecting a candidate or pressing Enter only previews it. **Use search result**
replaces species, A, q and Z together. Omitted ion A uses a stated isotope preset;
omitted q uses the fully stripped charge q=Z. Explicit invalid notation is
rejected. Ek and other independent inputs retain their values. ``P`` and ``N``
are phosphorus and nitrogen; lowercase ``p`` and ``n`` are proton and neutron.
The names/presets in ``PASS/tool/particles.py`` remain separate from mass data.
This module owns ``ParticleSpec``, A/Z/q validation, aliases and fuzzy lookup.
``PASS/tool/particle_masses.py`` loads the offline catalog and computes the
charge-dependent mass, including electron and ionization-energy corrections.
Both are independent of Qt and remain in ``PASS/tool`` as shared particle and
mass utilities. The Tools-specific calculation backends live alongside their
pages in ``PASS/gui``: ``beam_calculator.py`` (kinematics and power),
``optics_calculator.py`` (emittance and magnets), ``exciter_calculator.py``
(excitation preview), and ``rf_bucket.py`` (RF bucket). These backends remain
independent of Qt; Python callers use imports such as
``from PASS.gui.beam_calculator import solve_kinematics``. Their previous
``PASS.tool`` module paths have been removed. The move changes code organization
only; formulas, energy definitions and GUI behavior are unchanged.

Masses now use one fixed offline catalog, without a source selector or A*u
fallback. The runtime file is ``PASS/tool/mass_catalog.json``; its ``metadata``
records exact download URLs, hashes, and citations.
The former user ``PARTICLE_DATA.tsv`` is neither read nor packaged.

* `AME2020 unrounded table <https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt>`_
  supplies 3,558 ground-state nuclide records, primarily **neutral atomic mass**
  in u plus a free-neutron record; 1,008 carry extrapolation flags. Cite W. J. Huang et al.,
  Chinese Physics C 45, 030002 (2021), and M. Wang et al., 030003 (2021).
* `NIST CODATA 2022 <https://physics.nist.gov/cuu/Constants/Table/allascii.txt>`_
  supplies electron, muon, nucleon and light-nucleus masses and the u conversion.
  Proton, deuteron, triton, helion and alpha-particle ions use direct CODATA values.
  The proton mass is 938.27208943 MeV/c².
* `PDG 2026 <https://pdg.lbl.gov/2026/mcdata/mass_width_2026.txt>`_ supplies tau,
  pion and kaon masses. Particle and antiparticle use the corresponding same mass.
* `NIST ASD <https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html>`_ supplies
  successive ionization energies. The 2026-09-11 snapshot contains 6,019 stages,
  of which 172 have unknown energy. A missing required stage prevents that ion
  calculation. Original flags, reference codes and uncertainties are preserved.

The catalog is a formatted conversion of full official AME/CODATA/PDG downloads
and one ASD CSV export, not an official JSON release or individual-particle scrape.
H-minus affinity is separately cited from WebBook. Stable lookup IDs are retained,
but each record has readable identity fields and an original source line number.
For example, PDG ``20213`` now shows **a₁(1260)⁺**, charge +1, mass 1230 MeV/c²,
positive/negative errors 40 MeV/c² and width 420 MeV. The mass uncertainty of this
broad resonance is present in the official PDG table, not a conversion defect.

The file does **not** contain all possible isotopes or all charge-state masses.
AME covers its listed ground states; PDG stores 319 IDs with numeric masses, of
which the GUI exposes selected massive species. ASD stages stop at Z=110 in this
snapshot, and missing required stages prevent ion-mass calculation. Ionic masses
are computed on demand from neutral masses; nuclear isomers and negative ions
other than H-minus are unsupported. Source uncertainties and estimates are retained,
so authoritative provenance does not mean every entry has the same precision.

For positive ground-state ions the conversion is

.. math::

   m_{ion}c^2=M_{atom}[u]\,(u c^2)-q m_e c^2+\sum_{j=0}^{q-1} I_j.

Ionization work has a **positive** sign. Subtracting electrons alone would omit
binding effects. ASD values are element-level, without isotope-dependent shifts;
AME extrapolated masses are identified. Nuclear isomers are not represented.
No combined uncertainty is claimed without the needed source covariances and
isotope shifts. H-minus additionally uses the hydrogen electron affinity
0.754195(18) eV reported by Lykke et al. (1991), as cited in
`NIST WebBook <https://webbook.nist.gov/cgi/cbook.cgi?ID=C12385136&Mask=20>`_.
Other negative ions currently lack the required affinity correction. Unspecified
Z, absent isotopes and missing ionization energies produce explicit errors.

For this purpose the original evaluations are preferable to a general library:
`AtomDB <https://www.atomdb.org/>`_ targets X-ray plasma spectra;
`BODR <https://github.com/BlueObelisk/bodr>`_ collects chemical element/isotope data;
`iniabu <https://iniabu.readthedocs.io/en/latest/>`_ primarily exposes published
solar abundances. Standard atomic weights averaged over isotope abundances are
not the mass of a selected accelerator ion.

Kinematics, units and precision
---------------------------------

Ions and atoms use **AMeV**, meaning kinetic energy per integer nucleon count:
``Ek=K/A``. Electrons, muons, tau leptons and mesons (A=0) use **MeV per particle**.
Neutrons and antinucleons have A=1, so the per-nucleon and per-particle values agree.
Write ``D=max(A,1)`` for the common divisor. C-12 6+ at Ek=100 AMeV has complete
kinetic energy K=1200 MeV. Its evaluated mass is still charge dependent;
this convention does not approximate the rest mass by A*u.

The read-only, dimensionless mass ratio **mu** follows **Rest mass** in the
calculator results and the other tools' reference-beam panels. It updates with
particle/isotope/charge selection and is the numerical mass in u, not the integer
nucleon number A; invalid mass data clear the field.
For the same nuclide, changing charge leaves Z, N and A=Z+N unchanged, but changes
the electron count and binding energy, hence the actual rest mass and mu:

.. math::

   m_{q+1}c^2=m_qc^2-m_ec^2+I_q.

.. math::

   \mu=\frac{m_0}{u}=\frac{E_0}{uc^2},\qquad K=D E_k,\qquad D=\max(A,1).

For ions, the displayed total energy, momentum and rest mass are **E/A** (MeV),
**p/A** (GeV/c) and **m0/A** (MeV/c²). For A=0, the corresponding fields show
complete-particle E, p and m0. The mass ratio mu always refers to the complete
particle, remains visible after rest mass, and is not an energy divisor.
At fixed A and Ek, changing charge keeps complete kinetic energy K fixed,
but the evaluated mass correction changes gamma, beta and momentum, as well as
rigidity. Numeric results use 12 significant digits without asserting extra
measurement accuracy. Units follow names in parentheses.
The equations below use complete-particle K, E and p internally.

.. math::

   E=E_0+K,\quad \gamma=1+K/E_0,\quad
   \beta=\sqrt{1-\gamma^{-2}},\quad pc=\sqrt{K(K+2E_0)}.

Known Ek, displayed total energy/momentum, rigidity, beta or gamma can be inverted.
The beam calculator displays the magnitude ``Bρ=p/(abs(q)*e)``. Neutral particles
show no finite rigidity; selecting rigidity as their known quantity is rejected.
Ek=0 gives beta=0, gamma=1 and momentum=0. Invalid inputs clear dependent results.
The definition applies throughout Tools, including reference transfers, power,
RF, emittance, magnets and Exciter. Copied reference data use ``Ek_AMeV`` for
A>0 or ``Ek_MeV`` for A=0, with an explicit ``energy_normalization`` field.
The numerical ``solve_kinematics`` API takes normalized kinetic energy in eV,
but complete-particle total energy (eV) and momentum (eV/c) for inverse inputs.
Tracking APIs and their existing mass approximations are unchanged; match their
documented units and mass convention when transferring values.

Comparison with the original particle table
--------------------------------------------

Run ``python -m tests.codex.gui_tools.compare_mass_tables`` to reproduce the
read-only comparison with ``PASS/tool/PARTICLE_DATA.tsv``. It compares all
159,032 rows (3,047 nuclides, Z=1–92) by A/Z/q, using the same evaluated masses
as the GUI. All original rows are calculable. Results, a searchable offline HTML
viewer and full CSV files are written to
``tests/codex/gui_tools/artifacts/mass-comparison/``. Coverage files also list all
3,558 AME nuclides (511 absent from the TSV), 319 PDG entries and the additional
15 named GUI species. Original input hashes are recorded and neither mass file
is modified. ``test_mass_comparison.py`` independently checks every row using
decimal arithmetic and the tabulated atomic masses, electron mass and ionization energies.

Differences mean **current minus TSV**, in whole-particle rest mass, not mass
divided by A or mu. The median absolute difference is 32.424446 keV/c²; the
maximum is 1.886586 MeV/c² for Al-42 13+, whose AME mass is itself estimated
(atomic uncertainty about 0.500212 MeV/c²). The TSV's nearly constant charge-step
mass decrement omits the varying ionization correction. A different u conversion
scale and nuclide mass baseline are also visible. Its source version and
uncertainties are unknown, so these discrepancies are not certified measurement
errors or statistical significance estimates. The reports distinguish AME
estimates and original atomic uncertainties from combined ion-mass uncertainty.

Current, power and stored energy
---------------------------------

The former separate average/instantaneous forward/reverse entries are consolidated
into three physically distinct modes:

* **Current ↔ power**: choose known current (mA), kinetic-energy transport power
  (kW), or particle rate (1/s), and a common average/instantaneous time basis.
  The known input is not repeated in results. ``I=Ndot*abs(q)*e`` and
  ``P=Ndot*K[J]``; for charged particles ``P[kW]=I[mA]*D*Ek/abs(q)``
  with Ek in AMeV for A>0 or MeV for A=0 (``D=max(A,1)``).
  Circumference is unnecessary. Only particle-rate input works for neutral
  current/power conversion. A supplied peak is allowed; an average alone does
  not determine a peak. At zero Ek, power cannot uniquely determine current.
* **Pulsed beam**: real particles per pulse and actual extraction/repetition
  frequency give average current, average power and kinetic energy per pulse.
  Optional duration gives pulse-averaged current/power. These are peaks only
  for flat pulses; duration times repetition frequency must not exceed one.
* **Stored/circulating beam**: total real particles and circumference give
  ``f0=beta*c/C``, period, ``I=N*abs(q)*e*f0`` and stored kinetic energy ``N*K[J]``.
  Circulating current is not target power: the same particles circulate rather
  than being extracted every turn. Use pulsed delivery for actual extraction.

Zero current or count is valid. Power excludes rest energy and electrical
wall-plug consumption. Instantaneous current, Ek and power must refer to the
same cross section and instant; for a spread, use flux-weighted kinetic energy.

Tune diagram
------------

The condition is ``m*Qx+n*Qy=l`` with integer coefficients and order
``abs(m)+abs(n)``; m and n cannot both be zero. Full tune ranges may cross
integers or include negative values. Enable orders 1–12 independently and
filter single-plane, sum or difference resonances. Difference lines are
dashed. Custom lines have separate visibility and removal controls.

Coincident lines are reduced using the full triple (m,n,l), assigned to their
lowest geometric order, and drawn once. ``2*Qx=1`` remains second order;
``2*Qx=2`` is assigned to first order. Turning off a low order also removes its
coincident higher-order representations. Excessive enumeration is rejected
with a request to narrow the ranges or selected orders.

The default ranges are **9–10** on both axes, with one unnamed point,
**(9.47, 9.43)**. The plot fills the left area. The right pane holds
the ranges and **Working points** / **Resonance lines** tabs; drag the splitter
to adjust their widths. Double-click table cells to edit names and coordinates;
select a row to change its color and marker below the table. Toggle visibility
or add/remove rows; new names are blank. Names are optional, and whitespace-only
names are treated as blank. Visible points with nonempty names appear in a
compact legend inside the upper-right corner of the plot, using their own colors
and markers. Names are not written beside the points. Unnamed points remain
visible and selectable; hidden and out-of-range points do not enter the legend.
The resonance-order legend remains above the plot. Hover/select
a point to see coordinates. Out-of-range points are counted in the status
text. The Matplotlib toolbar provides zoom, pan and home. **Export plot** saves
SVG, PNG or PDF with the current theme and view.

**Paste** accepts CSV or tab-separated rows; **Import CSV** appends UTF-8
CSV/TSV. Columns are ``Qx,Qy`` or ``name,Qx,Qy[,color,marker]``, with optional
headers. Two-column rows receive blank names; an empty name column is also valid.
Markers are ``o``, ``s``, ``^``, ``D`` or ``+``. Invalid imports append
no rows. **Export CSV** includes all points, including hidden ones, with names,
coordinates, colors and markers, preserving blank names. Visibility is a local presentation choice.
The geometry does not calculate resonance strength or establish beam stability.

Reference inputs and exports
------------------------------

RF bucket, emittance, magnets and Exciter have independent reference particle
and Ek inputs. **Read beam calculator** copies a valid particle/species and Ek
as an explicit snapshot. Invalid source inputs do not replace the destination.
RF needs mass, charge and energy for its slip factor, bucket height and frequencies.
Emittance needs relativistic beta*gamma for normalization; magnets and Exciter
need rigidity and/or speed. Formula references describe each approximation.

Plots export SVG/PNG/PDF and CSV with explicit column units. Emittance and Exciter
place parameters and results together in the right vertical scrolling column.
The plot remains visible on the left. Formula windows list the local catalog path,
clickable official source URLs, and the shared per-nucleon/per-particle normalization equations.

RF bucket
----------

The navigation label is **RF bucket绘制**. Enter V, harmonic h, circumference C,
effective synchronous phase phi_s, and either gamma_t or eta. The reference
particle is retained. In this section E_r=E/D=m0*c²/D+Ek is in eV and q_r=abs(q)/D, with D=max(A,1). Defaults: 100 kV, h=4, C=100 m, phi_s=0 and gamma_t=6.

.. math::

   \eta=\gamma_t^{-2}-\gamma^{-2},\quad \phi=\phi_s-2\pi h z_{rel}/C,
   \quad \frac{d\phi}{dN}=2\pi h\eta\delta,
   \quad \frac{d\delta}{dN}=\frac{q_r V}{\beta^2E_r}(\sin\phi-\sin\phi_s).

These signs follow PASS RFCavity and the longitudinal Twiss drift. phi_s already
includes cavity offset and bunch-center phase. RF h is independent of the
Injection grouping harmonic. Stability requires eta*cos(phi_s)<0; for positive
eta a stable phase is 180 degrees. Zero charge, V, Ek or eta is rejected.

The positive Hamiltonian is ``H=pi*h*abs(eta)*delta²+U(theta)``, theta=phi-phi_s,
with ``U=sign(eta)*q_r*V/(beta²*E_r)*(cos(phi_s+theta)-cos(phi_s)+theta*sin(phi_s))``.
The lower adjacent saddle defines the separatrix. Inner contours and turning
points are solved numerically. Show phase or z_rel horizontally, delta or delta_E
vertically; ``delta_E≈beta²*E_r*delta``. The result tab gives half-heights, full
widths, area, Qs, fs, revolution/RF frequencies and energy gain per turn.
For A>0, energy heights and plots show ΔE/A (MeV), gains are per nucleon,
and bucket area uses eV s per nucleon. For A=0 these are per-particle quantities
in MeV and eV s. CSV energy columns explicitly use ``eV_per_nucleon`` or
``eV_per_particle``. The numeric core retains complete-particle energies,
divided by D at the display/export boundary.

.. math::

   Q_s=\sqrt{\frac{-h\eta q_r V\cos\phi_s}{2\pi\beta^2E_r}},\quad
   f_s=Q_s f_0,\quad f_0=\beta c/C,\quad f_{RF}=h f_0.

This is a frozen, single-harmonic, small-delta smooth model. It excludes collective
fields, radiation, multi-harmonic RF and capture ramps. Heights with ``|delta|>=1``
are rejected. It does not replace tracking. Background:
`CERN longitudinal beam dynamics <https://e-publishing.cern.ch/index.php/CYRSP/article/view/1586>`_.

Phase-space plotting and emittance calculation
------------------------------------------------

The tool **相空间绘制及发射度计算** starts with one independent parameter page.
Use **＋** to add a page or **复制当前页** to copy parameters, centroids, and visibility settings.
A copy receives a new color; nonblank legend names gain a “副本” suffix, while blank names remain blank.
Pages can be removed, but at least one remains. Page IDs remain stable after deletion.
All pages share the reference particle and energy at the top. Switching tabs changes only the editor;
enabled curves from all pages remain overlaid on one plot. No combined-beam emittance is calculated.

Each page enables **绘制相空间** (draw phase space) by default and disables
**绘制投影椭圆（含色散）** (draw projected ellipse with dispersion) by default.
The latter also requires the page's draw switch. Hidden pages still calculate results.
Pages receive different editable colors; betatron curves are solid and projected curves dashed.
The toolbar's **显示图例** (show legend) switch defaults to enabled. Each curve has its own editable name.
Empty or whitespace-only names omit that legend entry without hiding the curve; no legend box appears
when no entries remain. Tab labels are independent of legend names.
Centroids **x₀ (mm), x′₀ (mrad)** default to zero and translate both curves without changing
centered RMS statistics, covariance, emittance, or Twiss parameters.

Each page keeps parameters and results in the same scrolling column. Invalid input clears only that
page's results and curves, with an error marker on its tab; other valid pages continue to plot.
Image export preserves the displayed curves and legend. CSV exports both curves from every valid page,
including hidden curves, with ``page_id``, ``page_name``, legend names, and
``draw_betatron`` / ``draw_projected`` visibility flags. Coordinates include centroid offsets and use m and rad.
Invalid pages have no exported curve rows. When at least one page is valid, copied results include
all page inputs, valid results, and error reasons for invalid pages.

The unit **π·mm·mrad** uses the agreed area convention: input 1 means a geometric
RMS emittance of 1e-6 m rad in the formulas, with ellipse area pi*1e-6 m rad for
n=1. **Do not multiply the input by pi again.** No four-RMS factor is implied.
Normalization is ``epsilon_n=beta_rel*gamma_rel*epsilon`` with the same convention.

Twiss labels are **Twiss beta**, **Twiss alpha**, **Twiss gamma**. Gamma is in the
parameter block. In alpha/beta mode it updates automatically; beta/gamma mode
solves alpha and requires a positive/negative branch choice:

.. math::

   \gamma=\frac{1+\alpha^2}{\beta},\qquad \alpha=\pm\sqrt{\beta\gamma-1}.

Require beta>0 and beta*gamma>=1 for inversion. Gamma and beta cannot determine
the sign of alpha. These Twiss quantities are distinct from relativistic factors.

The one-plane model assumes uncorrelated betatron coordinates and momentum spread:

.. math::

   \sigma_x^2=\beta\epsilon+D^2\sigma_\delta^2,\quad
   \sigma_{x'}^2=\gamma\epsilon+D'^2\sigma_\delta^2,\quad
   \operatorname{Cov}(x,x')=-\alpha\epsilon+DD'\sigma_\delta^2.

The **projected** ellipse includes dispersive offsets of particles with different
momenta; its emittance is sqrt(det Sigma). At D=Dprime=0 it equals the betatron
ellipse. Cov is the average product of centered x and xprime, indicating their
joint tilt/correlation; ``r=Cov/(sigma_x*sigma_xprime)`` is dimensionless and lies
between -1 and 1 when defined. These do not represent another beam species.
Cov has units mm mrad, without an area-convention pi prefix.

Beam-size inversion subtracts the dispersion variance; an input below
``abs(D)*sigma_delta`` is inconsistent. At Ek=0 normalized-to-geometric inversion
is underdetermined. An n-sigma covariance ellipse has area pi*n²*epsilon; for a
nondegenerate 2D Gaussian it contains ``1-exp(-n²/2)``, about 39.35% at n=1.

The known-quantity selector retains geometric emittance, normalized emittance, and projected σx inputs,
and adds **投影 RMS 与相关性 → ε、Twiss** (projected RMS and correlation): enter σx, σxprime,
and either r or Cov(x,xprime). These inputs are projected, centered statistics including dispersion.
The calculation constructs the projected covariance and subtracts the dispersive contribution:

.. math::

   \Sigma=\begin{pmatrix}\sigma_x^2 & \operatorname{Cov}(x,x')\\
   \operatorname{Cov}(x,x') & \sigma_{x'}^2\end{pmatrix},\quad
   B=\Sigma-\sigma_\delta^2\begin{pmatrix}D^2 & DD'\\DD' & D'^2\end{pmatrix},

   \epsilon=\sqrt{\det B},\quad \beta=B_{11}/\epsilon,\quad
   \alpha=-B_{12}/\epsilon,\quad \gamma=B_{22}/\epsilon.

Both the projected matrix and B must be positive semidefinite; otherwise the inputs are inconsistent.
Correlation mode requires positive RMS values and ``|r|≤1``. A zero RMS requires covariance mode with Cov=0,
and the resulting r is displayed as undefined. At epsilon=0 the emittance and available statistics remain
visible, inferred Twiss values are undefined, and the contour can degenerate to a line or point.
The original emittance/beam-size modes retain their supplied Twiss values.
Results distinguish betatron and projected RMS values and emittances; projected results remain available
even when their curve is hidden. The reported Twiss parameters describe the betatron covariance.

Magnet conversion
-------------------

Choose particle-derived or directly entered **signed** B*rho=p/(q*e), positive
length L, and one known magnet quantity. Direct mode ignores disabled reference
inputs. Fields and strengths retain signs; beta/gamma normalization is not used
for the magnet coefficients.

* Dipole: ``k0=B/(Bρ)=1/rho``, ``theta=K0L=k0*L``, ``integral B dl=(Bρ)*theta``.
  L is effective arc length. Zero field has no finite radius.
* Quadrupole: ``k1=G/(Bρ)``, ``K1L=k1*L``, ``fx≈1/K1L``, ``fy≈-1/K1L``.
  G=dBy/dx. Focal lengths are thin-lens estimates.
* Sextupole and octupole: n=2 or 3, ``G_n=d^n By/dx^n``, ``k_n=G_n/(Bρ)``,
  ``K_nL=k_n*L``. At (x=r,y=0), ``By=G_n*r^n/n!``. Enter derivative, normalized
  strength, integrated strength/derivative, or field at a chosen reference radius.
  The 2! and 3! factors match PASS's actual kicks.
* Solenoid: ``Ks=Bz/(Bρ)``, ``kappa=Ks/2``, ``theta=kappa*L`` and
  ``integral Bz dl=(Bρ)*Ks*L``. The reference paraxial Larmor parameter has the
  sign used by PASS's rotation map. Focusing is kappa²; ``f≈1/(kappa²*L)`` is only
  a weak thin-lens estimate, not an exact finite-length focal distance.

Quadrupole, sextupole, octupole and solenoid default to normalized k1, k2, k3
and Ks respectively. For the three multipoles, enter the pole radius in mm:
``Bp[Gauss]=10000*(Bρ)*k_n*r[m]^n/n!`` with n=1,2,3. The signed ideal pole field
and its inverse are supported; its magnitude is abs(Bp). The solenoid instead
shows axial ``Bz[Gauss]=10000*(Bρ)*Ks``. Bore radius and Ks alone do not determine
an iron pole-face field; magnetic-circuit geometry and a field model are needed.
No fringe fields, saturation, hysteresis or coil-current calibration is included.

Exciter calculation and plotting
-----------------------------------

The page implements the four modes in ``PASS/commands/element/exciter.py``:
single_fm, single_fm_am, dual_fm, dual_fm_am. Inputs include plate voltage, gap,
effective plate length, circumference, tune or frequency, full sweep width,
period and the original dual-frequency/AM parameters. Switching frequency input
mode preserves the physical center frequency and width.

.. math::

   A_0=\frac{VL}{d\beta c|B\rho|},\quad f_c=Q_{excite}f_0,\quad
   \Delta f=\Delta Q f_0,\quad
   t_{arrive}=t_{0,start}+t_{elapsed}-\frac{z_{rel}+z_{center}}{\beta c}.

No z folding is performed. Only the waveform reduces arrival time modulo the
sweep period. A0 follows the current PASS charge-magnitude convention. Single FM
uses ``phi=2*pi*fc*tau+pi*df*tau*(tau-T)/T``. Dual FM uses the command's two phase
branches and ``2*cos(pi*df*tau/2)`` envelope; its amplitude can reach 2*A0.
The formula window lists both phase derivatives and AM equations explicitly.

AM updates by effective turn ``floor(t_elapsed*f0)`` and diverges at t_ext, so
AM plot windows must end before t_ext. The fixed-parameter preview plots the
external signal, not beam response, losses or emittance growth. Optional turn
markers evaluate arrival-phase sampling separately. Views include kick/envelope,
FM branches, AM factor and a one-sided Hann-window amplitude spectrum.

Numeric results include base kick, sampled peak/RMS, electric field, transit time,
frequencies, sampling rate and frequency-bin spacing. At least 24 samples per
bounded smooth frequency cycle are used, with a 200000-point cap; overly long
high-frequency windows are rejected. FM resets can generate additional broadband
content. FFT bin spacing is 1/window duration; a sampled peak is not a global
analytic bound. Waveform CSV columns use seconds, radians and Hz; spectrum CSV
uses frequency_Hz and kick_amplitude_rad. Parameters/results share a vertical
scrolling column and plots support the normal image/vector exports.

# Input and output of the model

## **1. Input Features (Predictors)**

These are the features fed _into_ the model. The model combines the visual information from the SEM image with the chemical/physical constraints defined by the PyBaMM parameters.

#### **A. Visual Input**

- **Parameter:** **3D Microstructure Volume**
- **Source:** 3D TIFF Grid (from `filename`).
- **Nature:** Tensor (Image Data).
- **Role:** Provides the torture, pore connectivity, and active material distribution.

#### **B. Physical & Electrochemical Design Constraints (Scalar Inputs)**

These 15 parameters define the chemistry and cell design applied to the microstructure. They are **Continuous** values taken directly from the Parquet file.

| #   | Parameter Name                            | Unit                 | Description                                                          |
| :-- | :---------------------------------------- | :------------------- | :------------------------------------------------------------------- |
| 1   | **SEI kinetic rate constant**             | $m \cdot s^{-1}$     | Rate of solid electrolyte interphase formation.                      |
| 2   | **Electrolyte diffusivity**               | $m^2 \cdot s^{-1}$   | How fast ions move through the liquid electrolyte.                   |
| 3   | **Initial concentration in electrolyte**  | $mol \cdot m^{-3}$   | Starting salt concentration.                                         |
| 4   | **Separator porosity**                    | $0-1$                | Porosity of the separator layer (distinct from the electrode image). |
| 5   | **Positive particle radius**              | $m$                  | Size of the active material particles (cathode).                     |
| 6   | **Negative particle radius**              | $m$                  | Size of the active material particles (anode).                       |
| 7   | **Positive electrode thickness**          | $m$                  | Physical depth of the cathode.                                       |
| 8   | **Negative electrode thickness**          | $m$                  | Physical depth of the anode.                                         |
| 9   | **Outer SEI solvent diffusivity**         | $m^2 \cdot s^{-1}$   | Transport rate of solvent through the SEI layer.                     |
| 10  | **Dead lithium decay constant**           | $s^{-1}$             | Rate at which active lithium becomes electrically isolated.          |
| 11  | **Lithium plating kinetic rate constant** | $m \cdot s^{-1}$     | Rate of metallic lithium plating on the anode.                       |
| 12  | **Neg. electrode LAM constant term**      | $s^{-1}$             | Loss of Active Material (LAM) proportional factor.                   |
| 13  | **Neg. electrode cracking rate**          | -                    | Rate of particle cracking in the anode.                              |
| 14  | **Outer SEI partial molar volume**        | $m^3 \cdot mol^{-1}$ | Volume expansion caused by SEI growth.                               |
| 15  | **SEI growth activation energy**          | $J \cdot mol^{-1}$   | Energy barrier for SEI growth reactions.                             |

---

### **2. Output Features (Targets)**

These are the values the model tries to predict. This section uses **Multi-Task Learning** weights to prioritize Cycle Life while ensuring the model learns the physical geometry.

#### **A. Electrochemical Performance (Primary Task)**

_These targets determine how long the battery lasts. The calculation logic solves the `NaN` issue by using the trend arrays._

| Parameter                              | Weight  | Nature     | Data Source & Calculation Logic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| :------------------------------------- | :------ | :--------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Projected Cycle Life (to 80%)**      | **1.0** | Regression | **Solves NaN issues:**<br>1. **Get True Nominal ($C_{nom}$):** Take `capacity_trend_ah[0]`.<br>2. **Define Limit:** $Limit = 0.8 \times C_{nom}$.<br>3. **Check Measured:** If any value in `capacity_trend_ah` $\le Limit$, use that cycle index.<br>4. **If Not Reached (Prediction):**<br> - Fit Linear Regression ($y = mx + c$) to the _last 50%_ of the trend array.<br> - **Solve for $x$:** $x_{pred} = (Limit - c) / m$.<br> - **Clamp:** If $m \ge 0$ (no fade) or $x_{pred} > 5000$, set target to 5000 (or max cycles). |
| **Capacity Fade Rate**                 | **0.8** | Regression | **The most stable metric:**<br>1. Perform Linear Fit on `capacity_trend_ah` vs `capacity_trend_cycles`.<br>2. **Target = Slope ($m$)**. <br>_(Represents degradation per cycle, e.g., -0.0005 Ah/cycle)._                                                                                                                                                                                                                                                                                                                           |
| **Initial DCIR (Internal Resistance)** | **0.5** | Regression | **Extract from `cycle_first` JSON:**<br>1. Identify the timestamp where current jumps from 0 (Rest) to Discharge.<br>2. Calculate $\Delta V = V_{rest} - V_{discharge\_start}$.<br>3. **Target = $\Delta V / I_{discharge}$**.                                                                                                                                                                                                                                                                                                      |
| **Capacity Retention (%)**             | **0.4** | Regression | **Calculation:**<br>$\frac{\text{Last Value in capacity\_trend\_ah}}{\text{First Value in capacity\_trend\_ah}} \times 100$.<br>_(Indicates health at the specific moment the sim stopped)._                                                                                                                                                                                                                                                                                                                                        |
| **Nominal Capacity**                   | **0.3** | Regression | **Dynamic Extraction:**<br>1. **Target = `capacity_trend_ah[0]`**.<br>_(Ignore the `nominal_capacity_Ah` column in the parquet as it is hardcoded to 5.0)._                                                                                                                                                                                                                                                                                                                                                                         |
| **Energy Density Indicator**           | **0.2** | Regression | **Average Voltage:**<br>1. From `cycle_first`, filter for Discharge phase only ($I < 0$).<br>2. **Target = `np.mean(Voltage_discharge)`**.<br>_(Higher average voltage = better energy density)._                                                                                                                                                                                                                                                                                                                                   |

#### **B. Microstructure Properties (Auxiliary Task)**

_These targets force the model to "understand" the geometry of the SEM image. We include Porosity here so the model learns to calculate it from the image pixels._

| Parameter                             | Weight  | Nature     | Data Source & Calculation Logic                                                                                              |
| :------------------------------------ | :------ | :--------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **Tortuosity Factor ($\tau$)**        | **0.3** | Regression | **Source:** `tau_factor` column.<br>_(Critical for ion transport efficiency)._                                               |
| **Electrode Porosity**                | **0.2** | Regression | **Source:** `porosity` (or `porosity_measured`) column.<br>_(The model must learn to segment pore vs solid from the image)._ |
| **Effective Diffusivity ($D_{eff}$)** | **0.2** | Regression | **Source:** `D_eff` column.<br>_(Derived from porosity and tortuosity)._                                                     |
| **Bruggeman Exponent ($b$)**          | **0.1** | Regression | **Source:** `bruggeman_derived` column.<br>_(Describes the relationship between porosity and tortuosity)._                   |

---

\ Model production model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 90 x_1 - 200 y_1 - 15 I_1 - 5000 H_1 - 8000 F_1 - 35 L_1 - 4800 P_1
   + 300 S_1 - 40 O_1 - 90 x_2 - 200 y_2 - 15 I_2 - 5000 H_2 - 8000 F_2
   - 35 L_2 - 4800 P_2 + 300 S_2 - 40 O_2 - 90 x_3 - 200 y_3 - 15 I_3
   - 5000 H_3 - 8000 F_3 - 35 L_3 - 4800 P_3 + 300 S_3 - 40 O_3 - 90 x_4
   - 200 y_4 - 15 I_4 - 5000 H_4 - 8000 F_4 - 35 L_4 - 4800 P_4 + 300 S_4
   - 40 O_4 - 90 x_5 - 200 y_5 - 15 I_5 - 5000 H_5 - 8000 F_5 - 35 L_5
   - 4800 P_5 + 300 S_5 - 40 O_5 - 90 x_6 - 200 y_6 - 15 I_6 - 5000 H_6
   - 8000 F_6 - 35 L_6 - 4800 P_6 + 300 S_6 - 40 O_6 - 1.35e+06 Constant
Subject To
 R0: I_0 = 15000
 R1: P_0 = 1000
 R2: L_0 = 0
 R3: S_0 = 0
 R4: I_6 >= 10000
 R5: H_0 = 0
 R6: F_0 = 0
 R7: e_0 = 0
 R8: x_0 = 0
 R9: y_0 = 0
 R10: z_0 = 0
 R11: O_0 = 0
 instance.demand_1: I_0 + x_1 + y_1 + e_1 = 20000
 inventory_1: I_0 + x_1 + y_1 - I_1 - S_1 = 0
 shortage1_1: e_1 - 1e+06 z_1 <= 0
 shortage2_1: - e_1 + 1e+06 z_1 <= 999999
 shortage3_1: - e_1 + L_1 + 1e+06 z_1 <= 1e+06
 shortage3_1: e_1 - L_1 + 1e+06 z_1 <= 1e+06
 sale_1: L_1 + S_1 = 20000
 employee_1: P_0 + H_1 - F_1 - P_1 = 0
 time_1: 5 x_1 - 160 P_1 - O_1 <= 0
 instance.overtime_upper_limit_1: - 20 P_1 + O_1 <= 0
 instance.demand_2: I_1 + x_2 + y_2 + e_2 = 40000
 inventory_2: I_1 + x_2 + y_2 - I_2 - S_2 = 0
 shortage1_2: e_2 - 1e+06 z_2 <= 0
 shortage2_2: - e_2 + 1e+06 z_2 <= 999999
 shortage3_2: - e_2 + L_2 + 1e+06 z_2 <= 1e+06
 shortage3_2: e_2 - L_2 + 1e+06 z_2 <= 1e+06
 sale_2: L_2 + S_2 = 40000
 employee_2: P_1 + H_2 - F_2 - P_2 = 0
 time_2: 5 x_2 - 160 P_2 - O_2 <= 0
 instance.overtime_upper_limit_2: - 20 P_2 + O_2 <= 0
 instance.demand_3: I_2 + x_3 + y_3 + e_3 = 42000
 inventory_3: I_2 + x_3 + y_3 - I_3 - S_3 = 0
 shortage1_3: e_3 - 1e+06 z_3 <= 0
 shortage2_3: - e_3 + 1e+06 z_3 <= 999999
 shortage3_3: - e_3 + L_3 + 1e+06 z_3 <= 1e+06
 shortage3_3: e_3 - L_3 + 1e+06 z_3 <= 1e+06
 sale_3: L_3 + S_3 = 42000
 employee_3: P_2 + H_3 - F_3 - P_3 = 0
 time_3: 5 x_3 - 160 P_3 - O_3 <= 0
 instance.overtime_upper_limit_3: - 20 P_3 + O_3 <= 0
 instance.demand_4: I_3 + x_4 + y_4 + e_4 = 35000
 inventory_4: I_3 + x_4 + y_4 - I_4 - S_4 = 0
 shortage1_4: e_4 - 1e+06 z_4 <= 0
 shortage2_4: - e_4 + 1e+06 z_4 <= 999999
 shortage3_4: - e_4 + L_4 + 1e+06 z_4 <= 1e+06
 shortage3_4: e_4 - L_4 + 1e+06 z_4 <= 1e+06
 sale_4: L_4 + S_4 = 35000
 employee_4: P_3 + H_4 - F_4 - P_4 = 0
 time_4: 5 x_4 - 160 P_4 - O_4 <= 0
 instance.overtime_upper_limit_4: - 20 P_4 + O_4 <= 0
 instance.demand_5: I_4 + x_5 + y_5 + e_5 = 19000
 inventory_5: I_4 + x_5 + y_5 - I_5 - S_5 = 0
 shortage1_5: e_5 - 1e+06 z_5 <= 0
 shortage2_5: - e_5 + 1e+06 z_5 <= 999999
 shortage3_5: - e_5 + L_5 + 1e+06 z_5 <= 1e+06
 shortage3_5: e_5 - L_5 + 1e+06 z_5 <= 1e+06
 sale_5: L_5 + S_5 = 19000
 employee_5: P_4 + H_5 - F_5 - P_5 = 0
 time_5: 5 x_5 - 160 P_5 - O_5 <= 0
 instance.overtime_upper_limit_5: - 20 P_5 + O_5 <= 0
 instance.demand_6: I_5 + x_6 + y_6 + e_6 = 18500
 inventory_6: I_5 + x_6 + y_6 - I_6 - S_6 = 0
 shortage1_6: e_6 - 1e+06 z_6 <= 0
 shortage2_6: - e_6 + 1e+06 z_6 <= 999999
 shortage3_6: - e_6 + L_6 + 1e+06 z_6 <= 1e+06
 shortage3_6: e_6 - L_6 + 1e+06 z_6 <= 1e+06
 sale_6: L_6 + S_6 = 18500
 employee_6: P_5 + H_6 - F_6 - P_6 = 0
 time_6: 5 x_6 - 160 P_6 - O_6 <= 0
 instance.overtime_upper_limit_6: - 20 P_6 + O_6 <= 0
Bounds
 e_0 free
 e_1 free
 e_2 free
 e_3 free
 e_4 free
 e_5 free
 e_6 free
 Constant = 1
Binaries
 z_0 z_1 z_2 z_3 z_4 z_5 z_6
Generals
 x_0 y_0 I_0 e_0 H_0 F_0 L_0 P_0 S_0 x_1 y_1 I_1 e_1 H_1 F_1 L_1 P_1 S_1
 x_2 y_2 I_2 e_2 H_2 F_2 L_2 P_2 S_2 x_3 y_3 I_3 e_3 H_3 F_3 L_3 P_3 S_3
 x_4 y_4 I_4 e_4 H_4 F_4 L_4 P_4 S_4 x_5 y_5 I_5 e_5 H_5 F_5 L_5 P_5 S_5
 x_6 y_6 I_6 e_6 H_6 F_6 L_6 P_6 S_6
End

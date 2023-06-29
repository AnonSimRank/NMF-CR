import numpy as np

def DecomAdjMat(adj_mat, weight_mat, lambda_0, lambda_1, max_iter_num, k, thre_val):

    row_num, col_num = adj_mat.shape
    A_tran = adj_mat.transpose()

    A_A_tran = np.matmul(adj_mat.A, A_tran.A)

    W_had_W = np.multiply(weight_mat, weight_mat)
    W2_had_A = np.multiply(W_had_W, adj_mat.A)

    #### initialize U,V,C ###################################
    rng = np.random.default_rng(seed=42)
    U = np.abs(rng.normal(loc=0, scale=1, size=(row_num, k)))
    V = np.abs(rng.normal(loc=0, scale=1, size=(row_num, k)))

    C0 = np.abs(rng.normal(loc=0, scale=1, size=(k, k)))
    C = (C0 + C0.transpose()) / 2
    ##########################################################

    i = 1
    while i <= max_iter_num:

        ######################## start update U ######################################

        W2_had_A_V = np.matmul(W2_had_A, V)

        A_tran_U = np.matmul(A_tran.A, U)
        A_A_tran_U = np.matmul(adj_mat.A, A_tran_U)

        C_tran_add_C = C.transpose() + C

        A_A_tran_U_C_tran_add_C = np.matmul(A_A_tran_U, C_tran_add_C)

        U_lr_numerator = 1 * W2_had_A_V + 1 * A_A_tran_U_C_tran_add_C

        U_V_tran = np.matmul(U, V.transpose())
        W2_had_U_V_tran = np.multiply(W_had_W, U_V_tran)

        W2_had_U_V_tran_V = np.matmul(W2_had_U_V_tran, V)

        U_tran_U = np.matmul(U.transpose(), U)
        U_C = np.matmul(U, C)
        U_C_tran = np.matmul(U, C.transpose())

        U_C_U_tran_U = np.matmul(U_C, U_tran_U)
        U_C_U_tran_U_C_tran = np.matmul(U_C_U_tran_U, C.transpose())

        U_C_tran_U_tran_U = np.matmul(U_C_tran, U_tran_U)
        U_C_tran_U_tran_U_C = np.matmul(U_C_tran_U_tran_U, C)

        U_lr_denominator = 1 * W2_had_U_V_tran_V + 1 * (U_C_U_tran_U_C_tran + U_C_tran_U_tran_U_C) + lambda_0 * U
        U_lr_denominator = np.maximum(U_lr_denominator, np.where(U_lr_denominator < 1.0e-16, 1, 0) * thre_val)

        U = (U_lr_numerator / U_lr_denominator) * U
        ######################### finish update U ######################################################################
        ##################################### start update V #########################################

        W2_had_A_tran = W2_had_A.transpose()
        W2_had_A_tran_U = np.matmul(W2_had_A_tran, U)

        V_lr_numerator = 1 * W2_had_A_tran_U

        U_V_tran_update = np.matmul(U, V.transpose())
        W2_had_U_V_tran_update = np.multiply(W_had_W, U_V_tran_update)
        W2_had_U_V_tran_update_tran = W2_had_U_V_tran_update.transpose()

        W2_had_U_V_tran_update_tran_U = np.matmul(W2_had_U_V_tran_update_tran, U)

        V_lr_denominator = 1 * W2_had_U_V_tran_update_tran_U + lambda_1 * V
        V_lr_denominator = np.maximum(V_lr_denominator, np.where(V_lr_denominator < 1.0e-16, 1, 0) * thre_val)

        V = (V_lr_numerator / V_lr_denominator) * V
        ##################################### finish update V ########################################
        ##################################### start update C ########################################

        A_tran_U = np.matmul(A_tran.A, U)
        U_tran_A_A_tran_U = np.matmul(A_tran_U.transpose(), A_tran_U)

        C_lr_numerator = 1 * U_tran_A_A_tran_U + 1 * C.transpose()

        U_tran_U = np.matmul(U.transpose(), U)
        U_tran_U_C = np.matmul(U_tran_U, C)
        U_tran_U_C_U_tran_U = np.matmul(U_tran_U_C, U_tran_U)

        C_lr_denominator = 1 * U_tran_U_C_U_tran_U + 1 * C
        C_lr_denominator = np.maximum(C_lr_denominator, np.where(C_lr_denominator < 1.0e-16, 1, 0) * thre_val)

        C = (C_lr_numerator / C_lr_denominator) * C
        ###################################### finish update C ######################################
        ###################################### compute loss function ################################

        U_update_V_tran_update = np.matmul(U, V.transpose())
        A_minus_U_update_V_tran_update = adj_mat.A - U_update_V_tran_update
        W_A_minus_U_update_V_tran_update = np.multiply(weight_mat, A_minus_U_update_V_tran_update)
        L_A = np.linalg.norm(W_A_minus_U_update_V_tran_update, 'fro') ** 2

        U_C_update = np.matmul(U, C)
        U_C_U_tran_update = np.matmul(U_C_update, U.transpose())
        A_A_tran_minus_U_C_U_tran_update = A_A_tran - U_C_U_tran_update
        L_A_A_tran = np.linalg.norm(A_A_tran_minus_U_C_U_tran_update, 'fro') ** 2

        C_minus_C_tran_update = C - C.transpose()
        L_C_C_tran = np.linalg.norm(C_minus_C_tran_update, 'fro') ** 2

        L_U = np.linalg.norm(U, 'fro') ** 2
        L_V = np.linalg.norm(V, 'fro') ** 2

        L = 1 / 2 * L_A + 1 / 2 * L_A_A_tran + 1 / 4 * L_C_C_tran + lambda_0 / 2 * L_U + lambda_1 / 2 * L_V
        print("The number of iterations is " + str(i) + "," + "loss value is " + str(L))
        ####################################### finish compute loss function ########################
        if i == max_iter_num or L < thre_val:
            return U, V, C
        else:
            i += 1



# R0 -> Loop counter
# R1 -> A
# R2 -> B
# R3 -> C
# R4 -> D
# R5 -> accumulator
# R6 -> temp
# Initialize const registers
MOVI R0, 99
LEA R1, A
LEA R2, B
LEA R3, C
LEA R4, D
# Loop and calculate
LOOP:
LD R5, R2, R0 # load B[i]
LD R6, R3, R0 # load C[i]
MUL R5, R5, R6 # B[i] * C[i]
LD R6, R4, R0 # load D[i]
ADD R5, R5, R6 # Add D[i] to B[i] * C[i]
RSHFA R5, R5, 1 # Div by 2
ST R5, R1, R0 # Write back to A[i]
ADD R0, R0, -1 # Decreament loop counter
BRGEZ LOOP # Loop if >= 0
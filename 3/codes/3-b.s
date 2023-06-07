SET Vst, 1
# First Run
SET Vln, 64
VLD, V0, B
VLD, V1, C
VMUL V0, V0, V1 # B * C
VLD V1, D
VADD V0, V0, V1 # (B * C) + D
VRSHFA V0, V0, 1 # ((B * C) + D) / 2
VST V0, A # Store to A
# Second run
SET Vln, 36
VLD, V0, B
VLD, V1, C
VMUL V0, V0, V1 # B * C
VLD V1, D
VADD V0, V0, V1 # (B * C) + D
VRSHFA V0, V0, 1 # ((B * C) + D) / 2
VST V0, A # Store to A
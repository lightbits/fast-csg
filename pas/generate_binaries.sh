# Compile .cu source file to .ptx
#   nvcc -ptx source.cu --gpu-architecture=sm_60 --use_fast_math
# Compile .ptx to cubin
#   ptxas --opt-level 1 --gpu-name sm_60 source.ptx --output-file source.cubin
# Disassemble cubin
#   nvdisasm -hex source.cubin > source.nvdisasm


ptxas --opt-level 3 --gpu-name sm_60 test4.ptx --output-file disasm_samples/test4.cubin && \
ptxas --opt-level 3 --gpu-name sm_60 test5.ptx --output-file disasm_samples/test5.cubin && \
ptxas --opt-level 3 --gpu-name sm_60 test6.ptx --output-file disasm_samples/test6.cubin && \
ptxas --opt-level 3 --gpu-name sm_60 test7.ptx --output-file disasm_samples/test7.cubin

nvdisasm -hex disasm_samples/test4.cubin > disasm_samples/test4.nvdisasm && \
nvdisasm -hex disasm_samples/test5.cubin > disasm_samples/test5.nvdisasm && \
nvdisasm -hex disasm_samples/test6.cubin > disasm_samples/test6.nvdisasm && \
nvdisasm -hex disasm_samples/test7.cubin > disasm_samples/test7.nvdisasm

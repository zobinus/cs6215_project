import os
import subprocess
import gtest.libgtest_toolbox as _C_toolbox
from typing import List, Literal, Optional

# single instance of the cuda binary utilities
_si_gtest_cuda_binary_utilities = None

# directory to metadata of SASS instruction (encrypted binary format)
# TODO(zhuobin): we need to change to system path
_metadata_dir = "/root/utils/sass_parser/output"

class cuda:
    @staticmethod
    def parse_sass(
        srcs : List[str],
        arch : List[int] = [90],
    ):
        output_dir = "/tmp"
        
        global _si_gtest_cuda_binary_utilities
        if not _si_gtest_cuda_binary_utilities:
            _si_gtest_cuda_binary_utilities = _C_toolbox.GWBinaryUtility_CUDA(_metadata_dir)

        # generate fatbin first
        cuda.nvcc(
            srcs=srcs,
            output_type="fatbin",
            arch=arch,
            lineinfo=True,
            output_dir=output_dir,
            args=[]
        )
        for src in srcs:
            src_name = os.path.splitext(os.path.basename(src))[0]
            fatbin_file_path = os.path.join(output_dir, f"{src_name}.fatbin")
            _si_gtest_cuda_binary_utilities.parse_fatbin(
                fatbin_file_path,   # fatbin_file_path
                output_dir          # dump_cubin_path
            )

        # readelf -x 16 /tmp/main.fatbin_0.cubin
        # cuobjdump -sass /tmp/main.fatbin

    @staticmethod
    def nvcc(
        srcs: List[str],
        output_type: Literal["ptx", "fatbin", "cubin"],
        arch: List[int],
        lineinfo: bool = False,
        output_dir: str = ".",
        args: Optional[List[str]] = None,
    ):  
        # make sure the output dir exists
        os.makedirs(output_dir, exist_ok=True)

        # compilation command
        base_cmd = ["nvcc"]

        # compile target map
        compile_target_map = {
            "ptx": "-ptx",
            "cubin": "--cubin",
            "fatbin": "--fatbin"
        }
        base_cmd.append(compile_target_map[output_type])

        # architecture
        if len(arch) == 0:
            raise ValueError("At least one architecture must be specified.")
        for arch_num in arch:
            base_cmd.extend(["--generate-code", f"arch=compute_{arch_num},code=sm_{arch_num}"])

        # whether embedding line infos inside the compiled binary
        if lineinfo:
            base_cmd.append("-lineinfo")

        # extra args
        if args:
            base_cmd.extend(args)

        # compile each source file
        for src in srcs:
            src_name = os.path.splitext(os.path.basename(src))[0]
            output_ext = output_type
            output_path = os.path.join(output_dir, f"{src_name}.{output_ext}")

            cmd = base_cmd + [src, "-o", output_path]

            print(f"execute command: {cmd}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode != 0:
                error_msg = (
                    f"Compilation failed for {src}:\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Error output:\n{process.stderr}"
                )
                raise RuntimeError(error_msg)
            

__all__ = [ "cuda" ]

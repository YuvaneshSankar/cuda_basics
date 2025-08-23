1) "The profiler returned an error code: 1 (0x1)
    The user does not have permission to access NVIDIA GPU Performance Counters on the target device."

        This usually happens because the NVIDIA driver restricts access to GPU performance counters by default for non-root users.


        Enable access to GPU performance counters:
                sudo nvidia-smi -pm 1
                sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

            Then allow all users to access performance counters:
                sudo nvidia-smi --compute-mode=DEFAULT
                sudo nvidia-smi --gom=0


        Set NVreg_RestrictProfilingToAdminUsers=0:
            Edit the NVIDIA driver configuration:
                sudo nano /etc/modprobe.d/nvidia.conf
            Add:
                options nvidia NVreg_RestrictProfilingToAdminUsers=0
            Update initramfs and reboot:
                sudo update-initramfs -u
                sudo reboot

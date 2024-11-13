#!/usr/bin/env python3

import pexpect
import subprocess
import shlex
import shutil

class Repl:
    def __init__(self):
        cargo = shutil.which("cargo")
        build_result = subprocess.run([cargo, "build"])
        assert build_result.returncode == 0
        self.pexpect = pexpect.spawn(cargo, ["run", "repl"], timeout=15)
        self.pexpect.expect('omclone> ', timeout=3)

    def run(self, cmd):
        if isinstance(cmd, list):
            cmd = shlex.join(cmd)
        self.pexpect.sendline(cmd)
        self.pexpect.expect('omclone> ')
        return self.pexpect.before

    def close(self):
        self.pexpect.close()

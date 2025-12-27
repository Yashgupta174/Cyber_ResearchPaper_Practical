# code/attacks/dos.py
import random

def dos_apply(packet, p_drop=0.2):
    """Return None if packet dropped (simulates DoS)."""
    if random.random() < p_drop:
        return None
    return packet

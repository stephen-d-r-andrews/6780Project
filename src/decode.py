import numpy as np, random, math

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," ","."]

K = len(alphabet)
char2idx = {c:i for i,c in enumerate(alphabet)}

P = np.loadtxt("data/letter_probabilities.csv", delimiter = ",")
M = np.loadtxt("data/letter_transition_matrix.csv", delimiter = ",")

eps  = 1e-8
P_s  = P + eps;      P_s /= P_s.sum()
M_s  = M + eps;      M_s /= M_s.sum(axis=0, keepdims=True)
logP = np.log(P_s)
logM = np.log(M_s)

def log_likelihood(perm, C):
    D = perm[C]
    return logP[D[0]] + logM[D[1:], D[:-1]].sum()

def propose(perm):
    # swap two *distinct* indices
    i, j = random.sample(range(K), 2)
    perm2 = perm.copy()
    perm2[i], perm2[j] = perm2[j], perm2[i]
    return perm2

def decode_nobreakpoint(ciphertext: str) -> str:
    C = np.array([char2idx[c] for c in ciphertext], dtype=int)

    perm     = np.random.permutation(K)
    ll_prev  = log_likelihood(perm, C)
    best     = perm.copy()
    best_ll  = ll_prev
    for _ in range(10000):
        cand    = propose(perm)
        ll_cand = log_likelihood(cand, C)

        # decode *current* perm by indexing through C
        decoded = "".join(alphabet[perm[idx]] for idx in C)

        # compute acceptance probability (overflow‐safe)
        diff  = ll_cand - ll_prev
        alpha = 1.0 if diff >= 0 else math.exp(diff)

        if random.random() < alpha: # accept new perm
            perm, ll_prev = cand, ll_cand
            # update best if improved
            if ll_cand > best_ll:
                best_ll, best = ll_cand, cand.copy()

    return "".join(alphabet[best[idx]] for idx in C), best_ll


def decode(ciphertext: str, has_breakpoint: bool) -> str:

    if has_breakpoint:
        """
        iterate through possible breakpoint locations
        """

        best_ll = -np.inf
        best_decoded = ""
        for i in range(1, len(ciphertext), 50):
            ciphertext1 = ciphertext[:i]
            ciphertext2 = ciphertext[i:]

            decode1, ll1 = decode_nobreakpoint(ciphertext1)
            decode2, ll2 = decode_nobreakpoint(ciphertext2)

            if ll1 + ll2 > best_ll:
                best_ll = ll1 + ll2
                best_decoded = "".join([decode1, decode2])

        return best_decoded
    else:
        return decode_nobreakpoint(ciphertext)[0]
    


def decode_break(ciphertext: str, has_breakpoint: bool) -> str:
    # no‐breakpoint case unchanged
    if not has_breakpoint:
        return decode_nobreakpoint(ciphertext)[0]

    n = len(ciphertext)

    # helper: compute combined log‐likelihood at split position i
    def combined_ll(i: int) -> float:
        _, ll1 = decode_nobreakpoint(ciphertext[:i])
        _, ll2 = decode_nobreakpoint(ciphertext[i:])
        return ll1 + ll2

    # binary search for the maximum of a unimodal discrete f(i)
    low, high = 1, n - 1
    while low < high:
        mid = (low + high) // 2

        # f(mid) vs f(mid+1)
        if combined_ll(mid) < combined_ll(mid + 1):
            # rising slope → max to the right
            low = mid + 1
        else:
            # falling (or peak) → max is at mid or to the left
            high = mid

    best_split = low

    # finally decode each segment at the found split
    left_dec,  _ = decode_nobreakpoint(ciphertext[:best_split])
    right_dec, _ = decode_nobreakpoint(ciphertext[best_split:])

    return left_dec + right_dec

def decode_test(ciphertext: str, has_breakpoint: bool) -> str:
    # no‐breakpoint case unchanged
    if not has_breakpoint:
        return decode_nobreakpoint(ciphertext)[0]




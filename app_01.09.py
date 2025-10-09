import streamlit as st
from functools import lru_cache
from typing import Optional

st.write("Hooray — we connected everything!")


@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed).

    This implementation uses memoization to avoid exponential recursion.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def format_result(n: int, value: Optional[int]) -> str:
    if value is None:
        return "No result"
    return f"F({n}) = {value}"


# streamlit UI
st.title("Fibonacci App")

# use an integer input and a small default
n = st.number_input("n (≥ 0)", min_value=0, value=10, step=1)

if st.button("Compute"):
    try:
        result = fibonacci(int(n))
        st.success(format_result(int(n), result))
    except Exception as e:
        st.error(f"Error computing Fibonacci: {e}")

if st.button("Celebrate"):
    st.balloons()

if st.button("Let it snow"):
    # Streamlit provides a fun `st.snow()` visual effect in newer versions.
    # If your Streamlit version doesn't include st.snow(), this will raise
    # an AttributeError — upgrade Streamlit (pip install -U streamlit)
    try:
        st.snow()
    except AttributeError:
        st.info("Your Streamlit version doesn't support `st.snow()`. Upgrade with: pip install -U streamlit")


st.button("fuck you")
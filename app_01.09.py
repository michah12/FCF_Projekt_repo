import streamlit as st

st.write ("Horray, we connected everything!")

# define fibonacci first
def fibonacci(n=0):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# streamlit UI
st.title("Fibonacci App")

n = st.number_input("n (≥ 0)", min_value=0, step=1)

if st.button("Compute"):
    result = fibonacci(int(n))
    st.write(f"F({n}) = {result}")

if st.button("Celebrate"):
    st.balloons()

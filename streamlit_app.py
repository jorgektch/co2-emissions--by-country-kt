import streamlit as st

#Título de lplicación
st.title('Mi primer app')
# Algo de texto
st.write("Esta aplicación tiene el objetivo de elevar cualquier número al cuadrado:")

#Comando para introducir número en Streamlit
x = st.number_input('Introduzca un número:')
#Mostrar el número al cuadrado
st.write('El número al cuadrado es:', x**2)

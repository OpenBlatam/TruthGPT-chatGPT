



continuar = "S":
while continuar == "s":
     n1 = int(input("Ingrese el primer número: "))
     ## Print (n1)
        n2 = int(input("Ingrese el segundo número: "))
        print ("+ SUMA")
        print ("- RESTA")
        print ("/ DIVISION")
        print ("* MULTIPLICACION")
        operacion = input("Dime la operacion que quieres hacer con estod dos numeros: ")
        print (operacion)
        if operacion == "+":
            resultado = n1 + n2
        elif operacion == "-":
            resultado = n1 - n2
        elif operacion == "/":
             resultado = n1 / n2
        elif operacion == "*":
              resultado = n1 * n2
       print ("E1 resultado de la operacion que elegiste es: ",resultado)
       continuar = input("Quieres hacer otra operacion? s/n: ")




   ## Segundo ejercicio

   vocales = ["A", "E", "I", "O", "U"]
   #print (vocales)
   palabra = input ("Dame una palabra: ")
   palabra = palabra.upper()
   print (palabra)
   for letra in palabra:
       print (letra)
   for vocal in vocales:
       print (vocal)
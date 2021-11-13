# My output in debug x64 mode:

elapsed time 0.066464 ms by float

elapsed time 0.071008 ms by double

relation float/double = 0.936
 
# My output in release x64 mode:

elapsed time 0.009216 ms by float

elapsed time 0.012000 ms by double

relation float/double = 0.768

Как мы видим разница вычисления чисел с float и double видна, при том заментнее она в release моде.

Вывод можно сделать такой, современные видеокарты предоставляют возможности быстрых вычислений с числами как и с единичной точностью, так и с двоичной.

Поэтому можно сделать вывод, для длительных вычислений, если повышенная точность вычислений не важна, лучше использовать float. 

Но если в решении задачи требуется избегать вычислительных ошибок, то используйте double.

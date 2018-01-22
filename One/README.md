
# Linear Regression
Finn en rett linje mellom datapunkt, som gir minst distanse fra alle punkt til linjen.

## Hypothesis h(x)
### if we know one feature
Linear function

`h(x) = 𝜭0 + 𝜭1 * x`

### If we know two features
`h(x) = 𝜭0 + 𝜭1 * x1 + 𝜭2 * x2 `

### If we know many features:
`h(x) = ∑(j=0 --> #f) 𝜭j * xj`

\#f = antall features && NB: x0 = 1

## Cost function

## Weights
### Weights function
`w = (Xt * X)^−1 * Xt * y`

`w` = den transponerte input ganget med input opphøyd i -1, ganget med den transponerte input gange output/klasse

### numpy matrix transpose
`np.matrix.transpose()`

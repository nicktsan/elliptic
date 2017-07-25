"""
Nicholas Tsang
V00805615

Program for obtaining a normalized elliptic transfer function and plotting 
its loss characteristic.

All equations are based off of the ones found in the textbook:
"Digital Signal Processing: Signals, Systems, and Filters" 
written by Andreas Antoniou.

Run this program by navigating to the containing directory with the command prompt, 
then type:
python elliptic.py
"""
import numpy as np
import math
import cmath
import scipy.special
import matplotlib.pyplot as plt

# method for calculating elliptic transfer function.
# k = selectivity, Ap = passband loss, Aa = minimum stopband loss
def elliptic(k, Ap, Aa):
	#calculate variables shown in equations 10.85 to 10.99 in the textbook
	kp = np.sqrt(1 - k**2)
	q0 = 0.5 * ((1 - np.sqrt(kp))/(1 + np.sqrt(kp)))
	q = q0 + 2*q0**5 + 15*q0**9 + 150*q0**13
	D = (10**(0.1*Aa) - 1)/(10**(0.1*Ap) - 1)
	n = math.ceil((np.log10(16*D))/np.log10(1/q)) # for numpy, log10 = log base 10
	r = int(n)/2
	A = (1/(2*n))*np.log((10**(0.05*Ap) + 1)/(10**(0.05*Ap) - 1)) # for numpy, log = ln()

	# for loop for calculating the convergent infinite series of the numerator 
	# in equation 10.92 in the textbook
	# this series rapidly coverges usually after 3 or 4 terms.
	sum0n = 0
	for m in range(0, 11):
		sum0n += ((-1)**m) * (q**(m*(m+1))) * np.sinh((2*m + 1)*A)
	
	# for loop for calculating the convergent infinite series of 
	# the denominator in equation 10.92 in the textbook
	# this series rapidly coverges usually after 3 or 4 terms.
	sum0d = 0
	for m in range(1, 11):
		sum0d += ((-1)**m) * (q**(m**2)) * np.cosh(2*m*A)

	o0 = np.absolute(((2*q**0.25) * sum0n)/(1 + 2 * sum0d))
	W = np.sqrt((1 + k*o0**2) * (1 + o0**2/k))
	om = np.zeros(r+1)
	for i in range(1, r+1):
		u = i
		if (n%2 == 0):
			u = i - 0.5
		# for loop for calculating the convergent infinite series of the 
		# numerator in equation 10.94 in the textbook
		# this series rapidly coverges usually after 3 or 4 terms.
		sum1n = 0
		for m in range(0, 11):
			sum1n += ((-1)**m) * (q**(m*(m+1))) * np.sin(((2*m + 1)*np.pi*u)/n)
		
		# for loop for calculating the convergent infinite series of the
		# denominator in equation 10.94 in the textbook
		# this series rapidly coverges usually after 3 or 4 terms.
		sum1d = 0
		for m in range(1, 11):
			sum1d += ((-1)**m) * (q**(m**2)) * np.cos((2*m*np.pi*u)/n)
		om[i] = ((2*q**0.25) * sum1n)/(1 + 2 * sum1d)

	V = np.zeros(r+1)
	for i in range(1, r+1):
		V[i] = np.sqrt((1 - (k*om[i]**2))*(1 - (om[i]**2/k)))

	# for a0 and b, coefficients start at index 1 instead of index 0. Index 0 is filler.
	a0 = np.zeros(r+1)
	b = np.zeros((2, r+1))
	for i in range(1, r+1):
		a0[i] = 1/(om[i]**2)
		b[0][i] = ((o0*V[i])**2 + (om[i]*W)**2)/(1 + o0**2 * om[i]**2)**2
		b[1][i] = (2 * o0 * V[i])/(1 + (o0**2 * om[i]**2))

	# for calculating the product of sequences in equation 10.99 of the textbook
	product_sequence = 1
	for i in range(1, r + 1):
		product_sequence *= b[0][i]/a0[i]

	H0 = o0 * product_sequence
	if (n%2 == 0):
		H0 = 10**(-0.05 * Ap) * product_sequence

	Aa_actual = 10*np.log10((10**(0.1*Ap) - 1)/(16*q**n) + 1)
	print "Order: %i" % int(n)
	print "Actual stopband loss: %f." % Aa_actual
	# for a0 and b, coefficients start at index 1 instead of index 0. Index 0 is filler.
	print "H0: %f" % H0
	if (n%2 == 0):
		print "D0(s): 1"
	else:
		print "D0(s): s + %f" % o0
	print "a0 coefficients: "
	print np.delete(a0, 0)
	print "b0 coefficients: "
	print np.delete(b[0], 0)
	print "b1 coefficients: "
	print np.delete(b[1], 0)
	print ""
	return H0, a0, b, o0, r, n

# variables are taken from Equation 10.85 of the textbook
def plot_loss(H0, a0, b, o0, r, n):
	# range of frequencies from 0 to 3, incrementing by 0.01 rad/s
	w = np.arange(0, 3.01, 0.01) 
	Aw = np.zeros(w.size) # loss

	# to convert from s domain to jw, s = w*1j
	for k in range(0, w.size):
		D0jw = 1 # variable to convert D0(s) to D0(jw)
		D0_neg_jw = 1 # variable to convert D0(s) to D0(-jw)
		if (n%2 != 0):
			D0jw = o0 + w[k]*1j
			D0_neg_jw = o0 + w[k]*-1j

		Hjw = H0/D0jw # this variable will hold the value for H(jw)
		H_negjw = H0/D0_neg_jw # this variable will hold the value for H(-jw)

		# for loop used to calculate the product_sequence in the transfer function
		for i in range(1, r+1):
			Hjw *= ((w[k]*1j)**2 + a0[i])/((w[k]*1j)**2 + b[1][i]*w[k]*1j + b[0][i])
			H_negjw *= ((w[k]*-1j)**2 + a0[i])/((w[k]*-1j)**2 + b[1][i]*w[k]*-1j + b[0][i])

		Lw2 = 1/(Hjw*H_negjw) # equation for L(w^2) = 1/(H(jw)H(-jw))
		Aw[k] = 10*np.log10(Lw2.real) # equation for A(w) = 10logL(w^2)

	plt.plot(w, Aw)
	plt.xlabel("w, rad/s")
	plt.ylabel("A(w), dB")
	plt.title("Loss Characteristics of Elliptic Transfer Function")
	plt.grid(True)
	plt.show()
	return 0


test0_H0, test0_a0, test0_b, test0_o0, test0_r, test0_n = elliptic(0.9, 0.1, 50.0) #even order
plot_loss(test0_H0, test0_a0, test0_b, test0_o0, test0_r, test0_n) #even order

test1_H0, test1_a0, test1_b, test1_o0, test1_r, test1_n = elliptic(0.7, 0.2, 61.0) #odd order
plot_loss(test1_H0, test1_a0, test1_b, test1_o0, test1_r, test1_n) #odd order

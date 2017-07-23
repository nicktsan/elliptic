"""
Nicholas Tsang
V00805615
Program for obtaining a normalized elliptic transfer function
"""
import numpy as np
import math
import scipy.special
import matplotlib.pyplot as plt

#method for calculating elliptic transfer function.
#k = selectivity, Ap = passband loss, Aa = minimum stopband loss
def elliptic(k, Ap, Aa):
	#calculate variables shown in equations 10.85 to 10.99 in the textbook
	kp = np.sqrt(1 - k**2)
	q0 = 0.5 * ((1 - np.sqrt(kp))/(1 + np.sqrt(kp)))
	q = q0 + 2*q0**5 + 15*q0**9 + 150*q0**13
	D = (10**(0.1*Aa) - 1)/(10**(0.1*Ap) - 1)
	n = math.ceil((np.log10(16*D))/np.log10(1/q))
	r = int(n)/2
	A = (1/(2*n))*np.log((10**(0.05*Ap) + 1)/(10**(0.05*Ap) - 1))

	sum0n = 0
	#for loop for calculating the convergent infinite series of the numerator in equation 10.92 in the textbook
	#this series rapidly coverges usually after 3 or 4 terms.
	for m in range(0, 11):
		sum0n += ((-1)**m) * (q**(m*(m+1))) * np.sinh((2*m + 1)*A)
	sum0d = 0
	#for loop for calculating the convergent infinite series of the denominator in equation 10.92 in the textbook
	#this series rapidly coverges usually after 3 or 4 terms.
	for m in range(1, 11):
		sum0d += ((-1)**m) * (q**(m**2)) * np.cosh(2*m*A)

	o0 = np.absolute(((2*q**0.25) * sum0n)/(1 + 2 * sum0d))
	
	W = np.sqrt((1 + k*o0**2) * (1 + o0**2/k))
	om = np.zeros(r+1)
	for i in range(1, r+1):
		u = i
		if (n%2 == 0):
			u = i - 0.5
		sum1n = 0
		#for loop for calculating the convergent infinite series of the numerator in equation 10.94 in the textbook
		#this series rapidly coverges usually after 3 or 4 terms.
		for m in range(0, 11):
			sum1n += ((-1)**m) * (q**(m*(m+1))) * np.sin(((2*m + 1)*np.pi*u)/n)
		sum1d = 0
		#for loop for calculating the convergent infinite series of the denominator in equation 10.94 in the textbook
		#this series rapidly coverges usually after 3 or 4 terms.
		for m in range(1, 11):
			sum1d += ((-1)**m) * (q**(m**2)) * np.cos((2*m*np.pi*u)/n)
		om[i] = ((2*q**0.25) * sum1n)/(1 + 2 * sum1d)
	V = np.zeros(r+1)
	for i in range(1, r+1):
		V[i] = np.sqrt((1 - (k*om[i]**2))*(1 - (om[i]**2/k)))
	#for a0 and b, coefficients start at index 1 instead of index 0. Index 0 is filler.
	a0 = np.zeros(r+1)
	b = np.zeros((2, r+1))
	for i in range(1, r+1):
		a0[i] = 1/(om[i]**2)
		b[0][i] = ((o0*V[i])**2 + (om[i]*W)**2)/(1 + o0**2 * om[i]**2)**2
		b[1][i] = (2 * o0 * V[i])/(1 + (o0**2 * om[i]**2))

	#for calculating the product of sequences in equation 10.99 of the textbook
	product_sequence = 1
	for i in range(1, r + 1):
		product_sequence *= b[0][i]/a0[i]
	H0 = o0 * product_sequence
	if (n%2 == 0):
		H0 = 10**(-0.05 * Ap) * product_sequence
	Aa_actual = 10*np.log10((10**(0.1*Ap) - 1)/(16*q**n) + 1)
	"""
	print kp
	print q0
	print q
	print D
	print n
	"""
	print "Actual stopband loss: %f." % Aa_actual
	#for a0 and b, coefficients start at index 1 instead of index 0. Index 0 is filler.
	print "a0 coefficients: "
	print np.delete(a0, 0)
	print "b0 coefficients: "
	print np.delete(b[0], 0)
	print "b1 coefficients: "
	print np.delete(b[1], 0)
	print "H0: %f" % H0
	
	return 0
#plot loss characteristics
def plot_loss(k, Ap, Aa):
	#variables of page 523 of 991 in textbook.
	wp = np.sqrt(k)
	wa = 1/np.sqrt(k)
	k1 = ((10**(0.1*Ap) - 1)/(10**(0.1*Aa) - 1))**0.5 #Eq 10.58
	#calculate variables shown in equations 10.85 to 10.99 in the textbook
	kp = np.sqrt(1 - k**2)
	q0 = 0.5 * ((1 - np.sqrt(kp))/(1 + np.sqrt(kp)))
	q = q0 + 2*q0**5 + 15*q0**9 + 150*q0**13
	D = (10**(0.1*Aa) - 1)/(10**(0.1*Ap) - 1)
	n = math.ceil((np.log10(16*D))/np.log10(1/q))
	#print n
	r = int(n)/2
	K = 1 #find out how to calculate this
	Fw = 0
	
	w = np.arange(0, 3.5, 0.5) #range of frequencies
	Aw = np.zeros(w.size) #loss
	for j in range(0, w.size):
		product_sequence = 1
		for i in range(1, r+1):
			if (n%2 != 0): #if odd order
				sn = scipy.special.ellipj(2*K*i/n, k)[0]
				Fw = (((-1)**r * w[j])/np.sqrt(k1)) * product_sequence
			else: #else is even order
				sn = scipy.special.ellipj(((2*i - 1)*K)/n, k)[0]
				Fw = ((-1)**r/np.sqrt(k1))*product_sequence
			om_i = np.sqrt(k)*sn
			product_sequence *= (w[j]**2 - om_i**2)/(1 - w[j]**2 * om_i**2)
		Lw2 = 1 + (10**(0.1*Ap) - 1) * Fw**2
		Aw[j] = 10*np.log10(Lw2)

	plt.plot(w, Aw)
	plt.xlabel("w, rad/s")
	plt.ylabel("A(w), dB")
	plt.title("Loss Characteristics of Elliptic Transfer Function")
	plt.grid(True)
	plt.show()
	return 0


#test0 = elliptic(0.9, 0.1, 50.0) #even order
#test1 = elliptic(0.7, 0.6, 40.0) #odd order
plot_loss(0.9, 0.1, 50.0) #even order
plot_loss(0.7, 0.2, 61.0) #odd order

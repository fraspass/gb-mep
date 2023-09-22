import numpy as np

def gamma_distance(lat_long_x, lat_long_y):
    lat_x, long_x = lat_long_x
    lat_y, long_y = lat_long_y
    r = 6365.079
    ret = 2*r*np.arcsin(np.sqrt(np.sin((lat_y-lat_x)/2)**2 + np.cos(lat_x)*np.cos(lat_y)*np.sin((long_y-long_x)/2)**2))
    return ret

def N_j(j, initial_times, t):
    """
    Compute the number of events that have started prior to time t for station j.

    Parameters
    ----------
    j : int
        The index of the station
    initial_times: np.array
        Matrix of start times for all stations. t[i, j] is the start time of the j-th event at station i.
    t : float
        The time at which the number of events is computed
    """
    initial_times_j = initial_times[j,:]

    return np.sum(initial_times_j <= t)

def N_j_prime(j, arrival_times, t):
    """
    Compute the number of events that have ended prior to time t for station j.

    Parameters
    ----------
    j : int
        The index of the station
    arrival_times: np.array
        Matrix of exit times for all stations. t[i, j] is the exit time of the j-th event at station i.
    t : float
        The time at which the number of events is computed
    """
    arrival_times_j = arrival_times[j,:]

    return np.sum(arrival_times_j <= t)

def compensator(i, t_compensator, initial_times, arrival_times, lams, kappa, kappa_prime, alphas, betas, alpha_primes, beta_primes, gamma_matrix):
    """
    Compute the compensator

    Parameters
    ----------
    i : int
        The index of the station
    t_compensator : float
        The time at which the compensator is computed
    initial_times: np.array
        The vector of initial times for all stations
    arrival_times : np.array
        The vector of arrival times for all stations
    lambdas : np.array
        The vector of lambdas
    kappa : function
        The kernel function
    kappa_prime : function
        The second kernel function
    alphas : np.array
        The vector of alpha parameters
    betas : np.array
        The vector of beta parameters
    alpha_primes : np.array
        The vector of alpha prime parameters
    beta_primes : np.array
        The vector of beta prime parameters
    gamma_matrix : np.array
        The matrix of gamma distances
    """

    lam_i = lams[i]
    alpha_i = alphas[i]
    beta_i = betas[i]
    alpha_prime_i = alpha_primes[i]
    beta_prime_i = beta_primes[i]
    # M is the number of stations
    M = len(lams)
    for j in range(M):
        kappa_ij = kappa(gamma_matrix[i, j])
        kappa_prime_ij = kappa_prime(gamma_matrix[i, j])
        multiplicative_term_one = kappa_ij*alpha_i/beta_i
        multiplicative_term_two = kappa_prime_ij*alpha_prime_i/beta_prime_i
        summation_one = 0
        summation_two = 0
        for k in range(N_j(j, initial_times, t_compensator)):
            summation_one += np.exp(-beta_i*(t_compensator-initial_times[j, k])) - 1

        for h in range(N_j_prime(arrival_times, t_compensator)):
            summation_two += np.exp(-beta_prime_i*(t_compensator-arrival_times[h, k])) - 1

    return lam_i*t_compensator + multiplicative_term_one*summation_one + multiplicative_term_two*summation_two

def CIF(i, k, initial_times, arrival_times, lams, kappa, kappa_prime, alphas, betas, alpha_primes, beta_primes, gamma_matrix):
    """
    Compute the conditional intensity function when the intensity kernel is exponential
    """
    t_ik = initial_times[i, k]
    lam_i = lams[i]
    alpha_i = alphas[i]
    beta_i = betas[i]
    alpha_prime_i = alpha_primes[i]
    beta_prime_i = beta_primes[i]
    M = len(lams)
    kappa_ij = kappa(gamma_matrix)
    kappa_prime_ij = kappa_prime(gamma_matrix)

    def A(j, k):
        if k==1:
            summation = 0
            for l in range(N_j(j, initial_times, initial_times[i, 1])):
                summation += np.exp(-beta_i*(initial_times[i, 1]-initial_times[j, l]))
            return summation
        else:
            second_term = 0
            for l in range(N_j(j, initial_times, initial_times[i, k-1] + 1, N_j(j, initial_times, initial_times[i, k]) + 1)):
                second_term += np.exp(-beta_i*(initial_times[i, k]-initial_times[j, l]))

            return A(j, k-1)*np.exp(-beta_i*(initial_times[i, k]-initial_times[i, k-1])) + second_term 
        
    def A_prime(j, k):
        if k==1:
            summation = 0
            for l in range(N_j_prime(j, arrival_times, initial_times[i, 1])):
                summation += np.exp(-beta_i*(initial_times[i, 1]-arrival_times[j, l]))
            return summation
        else:
            second_term = 0
            for l in range(N_j_prime(j, arrival_times, initial_times[i, k-1] + 1, N_j_prime(j, arrival_times, initial_times[i, k]) + 1)):
                second_term += np.exp(-beta_i*(initial_times[i, k]-arrival_times[j, l]))

            return A_prime(j, k-1)*np.exp(-beta_prime_i*(initial_times[i, k]-initial_times[i, k-1])) + second_term 
    principal_summation = 0
    for j in range(M):
        principal_summation += kappa_ij[i, j] * alpha_i * A(j, k) + kappa_prime_ij[i, j] * alpha_prime_i * A_prime(j, k)
    return lam_i + principal_summation

        
    
                
        
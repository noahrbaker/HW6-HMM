import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """

        if not isinstance(observation_states, np.ndarray):
            raise TypeError("observation_states must be a numpy array")
        if not isinstance(hidden_states, np.ndarray):
            raise TypeError("hidden_states must be a numpy array")
        if not isinstance(prior_p, np.ndarray):
            raise TypeError("prior_p must be a numpy array")
        if not isinstance(transition_p, np.ndarray):
            raise TypeError("transition_p must be a numpy array")
        if not isinstance(emission_p, np.ndarray):
            raise TypeError("emission_p must be a numpy array")
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        

        if not isinstance(input_observation_states, np.ndarray):
            raise TypeError("input_observation_states must be a numpy array")
        assert input_observation_states.shape[0] > 0, "input_observation_states must be of length greater than 0"
        
        # Step 1. Initialize variables
        len_hid = self.hidden_states.shape[0]
        len_obs = input_observation_states.shape[0]
        forward_mat = np.zeros((len_hid, len_obs))
        
        # Step 2. Calculate probabilities
        # Initialize the first column
        for i in range(len_hid):
            forward_mat[i, 0] = self.prior_p[i] * self.emission_p[i, self.observation_states_dict[input_observation_states[0]]]

        # Continue with the other columns
        for t in range(1, len_obs):
            for j in range(len_hid):
                tran_sum = np.sum(forward_mat[:, t-1] * self.transition_p[:, j])
                emis_sum = self.emission_p[j, self.observation_states_dict[input_observation_states[t]]]
                forward_mat[j, t] = tran_sum * emis_sum

        # Step 3. Return final probability
        forward_probability = np.sum(forward_mat[:, -1])
        return forward_probability




    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """     
        
        if not isinstance(decode_observation_states, np.ndarray):
            raise TypeError("decode_observation_states must be a numpy array")
        assert decode_observation_states.shape[0] > 0, "decode_observation_states must be of length greater than 0"

        # Initialize the vars for indexing   
        len_obs = len(decode_observation_states)
        len_hid = len(self.hidden_states)
        
        # Convert observation states to indices to use later
        obs_indices = np.array([self.observation_states_dict[obs] for obs in decode_observation_states])
        
        # Make empty arrays
        viterbi_mat = np.zeros((len_obs, len_hid))
        best_path = np.zeros((len_obs, len_hid), dtype=int)
        
        # Initialize first step
        viterbi_mat[0] = np.log(self.prior_p) + np.log(self.emission_p[:, obs_indices[0]])
        
        # Recursion step
        for t in range(1, len_obs):
            for j in range(len_hid):
                prob = viterbi_mat[t-1] + np.log(self.transition_p[:, j]) + np.log(self.emission_p[j, obs_indices[t]])
                viterbi_mat[t, j] = np.max(prob)
                best_path[t, j] = np.argmax(prob)
        
        # Traceback
        best_hidden_state_sequence = np.zeros(len_obs, dtype=int)
        best_hidden_state_sequence[-1] = np.argmax(viterbi_mat[-1])
        
        for t in range(len_obs-2, -1, -1):   # second to last, moving backwards
            best_hidden_state_sequence[t] = best_path[t+1, best_hidden_state_sequence[t+1]]

        # Have it be in text, and not in index form
        return [self.hidden_states[i] for i in best_hidden_state_sequence]

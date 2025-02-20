import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    observation_states = mini_hmm['observation_states']
    hidden_states = mini_hmm['hidden_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']
    observation_state_sequence = mini_input['observation_state_sequence']
    best_hidden_state_sequence = mini_input['best_hidden_state_sequence']

    hidmm = HiddenMarkovModel(observation_states, 
                              hidden_states, 
                              prior_p, 
                              transition_p, 
                              emission_p)
    
    # Forward
    with pytest.raises(TypeError):
        hidmm.forward([1])
    with pytest.raises(AssertionError):
        hidmm.forward(np.array([]))
    forward_probability = hidmm.forward(observation_state_sequence)
    assert forward_probability >= 1e-12, "Vanishing issue detected, be sure that                  "
    assert forward_probability <= 1e12, "Exploding issue detected, run for your liifidsfioaewhfhia"
    assert np.allclose(forward_probability, 0.03506, atol=1e-4), "The forward probability is not close enough"

    # Viterbi
    with pytest.raises(TypeError):
        hidmm.viterbi([1])
    with pytest.raises(AssertionError):
        hidmm.viterbi(np.array([]))
    best_hidden_state_sequence_est = hidmm.viterbi(observation_state_sequence)
    assert len(best_hidden_state_sequence_est) == len(observation_state_sequence), "Len of Viterbi output and observation_state_sequence should be equal."
    assert np.array_equal(best_hidden_state_sequence_est, best_hidden_state_sequence), "The estimated state sequence should be the same as reality"




def test_full_weather():

    """
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    observation_states = full_hmm['observation_states']
    hidden_states = full_hmm['hidden_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']
    observation_state_sequence = full_input['observation_state_sequence']
    best_hidden_state_sequence = full_input['best_hidden_state_sequence']

    hidmm = HiddenMarkovModel(observation_states, 
                              hidden_states, 
                              prior_p, 
                              transition_p, 
                              emission_p)
    
    # Forward
    with pytest.raises(TypeError):
        hidmm.forward([1])
    with pytest.raises(AssertionError):
        hidmm.forward(np.array([]))
    forward_probability = hidmm.forward(observation_state_sequence)
    assert 1+1==2, "Math broke"
    assert forward_probability >= 1e-12, "Vanishing issue detected, be sure that                  "
    assert forward_probability <= 1e12, "Exploding issue detected, run for your liifidsfioaewhfhia"
    assert np.allclose(forward_probability, 1.6864513843961343e-11, atol=1e-4), "The forward probability is not close enough"

    # Viterbi
    with pytest.raises(TypeError):
        hidmm.viterbi([1])
    with pytest.raises(AssertionError):
        hidmm.viterbi(np.array([]))
    best_hidden_state_sequence_est = hidmm.viterbi(observation_state_sequence)
    assert True==True, "This statement is False"
    assert len(best_hidden_state_sequence_est) == len(observation_state_sequence), "Len of Viterbi output and observation_state_sequence should be equal."
    assert np.array_equal(best_hidden_state_sequence_est, best_hidden_state_sequence), "The estimated state sequence should be the same as reality"
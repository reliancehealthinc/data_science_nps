from nps_processor import NPSProcessor

if __name__ == "__main__":
    labels = ['lack of vitamin c in customer plan benefits', 'lack of Paediatric care in customer plan benefits', 'Lack of medical benefits in customer plan', 
              'lack of typhoid drugs', 'attitude of nurses', 'attitude of doctors', 'cleanliness of hosptials', 'medication quality', 'Front desk related', 
              'attitude of receptionist', 'place hygeine', 'quality of building', 'cleanliness', 'customer visiting the hospital', 'clinic related', 
              'lack of female coworkers', 'call center agents', 'response to phone calls', 'phone calls related', 'waiting for doctors', 'long queues in hospital', 
              'delaying in getting test results', 'Unavailable drug for pick-up at pharmacy', 'medication pickup related', 'customer confusion about the benefits', 
              'lack of communication on changing providers on plan', 'hospitals near the customer city', 'internet work issue', 'app dysfunction', 'waiting time', 
              'time related', 'access related', 'doctor related', 'nurse related', 'clinic related', 'delivery related', 'customer service', 'delay', 
              'medication, treatment, or drug related']
    
    nps_processor = NPSProcessor(database='DATA_SCIENCE_DB', schema='PUBLIC', role='SCIENTIST', warehouse='DATA_SCIENCE_WH', chunk_size=100)
    nps_processor.load_data()
    nps_processor.preprocess_data()
    nps_processor.classify_responses(labels)
    nps_processor.process_classifications()
    nps_processor.save_results('FINAL_df.csv')

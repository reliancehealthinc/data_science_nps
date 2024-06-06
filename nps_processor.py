import pandas as pd
from transformers import pipeline
from snowflake_utils import get_snowflake_connection, load_data, upload_data

class NPSProcessor:
    def __init__(self, database, schema, role, warehouse, chunk_size=100):
        """
        Initialize the NPSProcessor with Snowflake connection details and setup the classifier.

        Parameters:
        database (str): The name of the database.
        schema (str): The schema to be used.
        role (str): The role to be used.
        warehouse (str): The warehouse to be used.
        chunk_size (int): The number of rows to process in each chunk. Default is 100.
        """
        print("Initializing NPSProcessor...")
        self.connection = get_snowflake_connection(database=database, schema=schema, role=role, warehouse=warehouse)
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device='mps')
        self.chunk_size = chunk_size
        print("NPSProcessor initialized.")
       
    def load_data(self):
        """
        Load data from Snowflake into DataFrames.
        """
        print("Loading data from Snowflake...")
        self.provider_ranking_nps = load_data(self.connection, "SELECT * FROM DATA_SCIENCE_DB.PUBLIC.PROVIDER_RANKING_FINAL")
        self.nps_data = load_data(self.connection, "SELECT * FROM DATA_SCIENCE_DB.PUBLIC.NPS_PROVIDER_RAW")
        print("Data loading completed.")
    
    def preprocess_data(self):
        """
        Preprocess the NPS data by merging relevant columns into a single customer response column.
        """
        print("Preprocessing data...")
        def merge_columns(row):
            return ' '.join(str(cell) for cell in row if not pd.isna(cell))
        
        self.nps_data['CUSTOMER_RESPONSE'] = self.nps_data[
            ['we_are_sorry_that_you_had_an_unpleasant_experience._can_you_please_provide_more_information_on_why_you_gave_that_rating?',
             'we_are_glad_you_had_a_great_experience._can_you_please_provide_more_information_on_why_you_gave_that_rating?',
             'thank_you_for_your_response._can_you_please_provide_more_information_on_why_you_gave_that_rating?']
        ].apply(merge_columns, axis=1)
        
        self.nps_data.drop(
            columns=['we_are_sorry_that_you_had_an_unpleasant_experience._can_you_please_provide_more_information_on_why_you_gave_that_rating?',
                     'we_are_glad_you_had_a_great_experience._can_you_please_provide_more_information_on_why_you_gave_that_rating?',
                     'thank_you_for_your_response._can_you_please_provide_more_information_on_why_you_gave_that_rating?'], 
            inplace=True
        )
        
        self.nps_data_coi = self.nps_data[['CUSTOMER_RESPONSE', 'provider_name']]
        self.provider_ranking_nps_coi = self.provider_ranking_nps[['provider_name', 'type_service', 'address']]
        self.nps_data_qa = pd.merge(self.provider_ranking_nps_coi, self.nps_data_coi, on='provider_name').drop_duplicates()
        print("Preprocessing completed.")
    
    def classify_responses(self, labels):
        """
        Classify customer responses using zero-shot classification.

        Parameters:
        labels (list): A list of labels for classification.
        """
        print("Classifying responses...")
        detailed_scores = []
        total_rows = len(self.nps_data_qa)
        for start in range(0, total_rows, self.chunk_size):
            chunk = self.nps_data_qa.iloc[start:start + self.chunk_size]
            for i, row in chunk.iterrows():
                input_text = row['CUSTOMER_RESPONSE']
                if input_text.strip():
                    try:
                        model_dict = self.classifier(input_text, labels, multi_label=True)
                        result_dict = dict(zip(model_dict.get('labels'), model_dict.get('scores')))
                        score_dict = {'CUSTOMER_RESPONSE': input_text}
                        score_dict.update(result_dict)
                        detailed_scores.append(score_dict)
                    except Exception as e:
                        print(f"An error occurred at index {i} with text: {input_text}. Error: {e}")
        self.detailed_scores_df = pd.DataFrame(detailed_scores)
        print("Classification for chunk completed.")
    
    def process_classifications(self):
        """
        Process the classified scores into structured categories and integrate them into the main data.
        """
        print("Processing classifications...")
        self.detailed_scores_df['Health_benefits_coverage'] = ((self.detailed_scores_df['lack of vitamin c in customer plan benefits'] > 0.7).astype(int) |
                                                               (self.detailed_scores_df['lack of Paediatric care in customer plan benefits'] > 0.9).astype(int) |
                                                               (self.detailed_scores_df['Lack of medical benefits in customer plan'] > 0.9).astype(int) |
                                                               (self.detailed_scores_df['lack of typhoid drugs'] > 0.9).astype(int))
        
        self.detailed_scores_df['provider_quality'] = ((self.detailed_scores_df['attitude of nurses'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['attitude of doctors'] > 0.8).astype(int) |
                                                       (self.detailed_scores_df['cleanliness of hosptials'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['medication quality'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['Front desk related'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['attitude of receptionist'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['place hygeine'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['quality of building'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['cleanliness'] > 0.99).astype(int) |
                                                       (self.detailed_scores_df['customer visiting the hospital'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['clinic related'] > 0.95).astype(int) |
                                                       (self.detailed_scores_df['lack of female coworkers'] > 0.95).astype(int))
        
        self.detailed_scores_df['RCC_quality'] = ((self.detailed_scores_df['call center agents'] > 0.9).astype(int) |
                                                  (self.detailed_scores_df['response to phone calls'] > 0.95).astype(int) |
                                                  (self.detailed_scores_df['phone calls related'] > 0.9).astype(int))
        
        self.detailed_scores_df['provider_wait_times'] = ((self.detailed_scores_df['waiting for doctors'] > 0.9).astype(int) |
                                                          (self.detailed_scores_df['long queues in hospital'] > 0.9).astype(int) |
                                                          (self.detailed_scores_df['delaying in getting test results'] > 0.99).astype(int))
        
        self.detailed_scores_df['medication_related'] = ((self.detailed_scores_df['Unavailable drug for pick-up at pharmacy'] > 0.9).astype(int) |
                                                         (self.detailed_scores_df['medication pickup related'] > 0.9).astype(int))
        
        self.detailed_scores_df['customer_education'] = ((self.detailed_scores_df['customer confusion about the benefits'] > 0.95).astype(int) |
                                                         (self.detailed_scores_df['lack of communication on changing providers on plan'] > 0.95).astype(int))
        
        self.detailed_scores_df['provider_network'] = (self.detailed_scores_df['hospitals near the customer city'] > 0.95).astype(int)
        self.detailed_scores_df['tech_issues'] = ((self.detailed_scores_df['internet work issue'] > 0.95).astype(int) |
                                                  (self.detailed_scores_df['app dysfunction'] > 0.95).astype(int))
        
        self.detailed_scores_df['wait_time_related'] = (self.detailed_scores_df['waiting time'] > 0.99).astype(int)
        self.detailed_scores_df['time related'] = (self.detailed_scores_df['time related'] > 0.99).astype(int)
        self.detailed_scores_df['access related'] = (self.detailed_scores_df['access related'] > 0.99).astype(int)
        self.detailed_scores_df['doctor related'] = (self.detailed_scores_df['doctor related'] > 0.99).astype(int)
        self.detailed_scores_df['nurse related'] = (self.detailed_scores_df['nurse related'] > 0.99).astype(int)
        self.detailed_scores_df['clinic related'] = (self.detailed_scores_df['clinic related'] > 0.99).astype(int)
        self.detailed_scores_df['delivery related'] = (self.detailed_scores_df['delivery related'] > 0.98).astype(int)
        self.detailed_scores_df['customer service related'] = (self.detailed_scores_df['customer service'] > 0.99).astype(int)
        self.detailed_scores_df['delay'] = (self.detailed_scores_df['delay'] > 0.99).astype(int)
        self.detailed_scores_df["medication, treatment, or drug related"] = (self.detailed_scores_df["medication, treatment, or drug related"] > 0.9).astype(int)

        self.detailed_scores_df.loc[(self.detailed_scores_df['medication, treatment, or drug related'] == 1) & (self.detailed_scores_df['delivery related'] == 0), 'provider_quality'] = 1
        self.detailed_scores_df.loc[(self.detailed_scores_df['medication_related'] == 1) & (self.detailed_scores_df['delivery related'] == 0), 'provider_quality'] = 1

        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("customer service", case=False, na=False), 'RCC_quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("mosquitoes", case=False, na=False), 'provider_quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("during my visit", case=False, na=False), 'provider_quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("during my visit", case=False, na=False), 'RCC_quality'] = 0
        self.detailed_scores_df.loc[self.detailed_scores_df['doctor related'] == 1, 'provider_quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("gym", case=False, na=False), 'provider_quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("gym", case=False, na=False), 'RCC_quality'] = 0
        self.detailed_scores_df.loc[(self.detailed_scores_df['access related'] == 1) & (self.detailed_scores_df['time related'] == 1), 'provider_wait_times'] = 1
        self.detailed_scores_df.loc[(self.detailed_scores_df['delay'] == 1) & (self.detailed_scores_df['RCC_quality'] == 0), 'provider_wait_times'] = 1

        self.detailed_scores_df['Medical_Staff'] = ((self.detailed_scores_df['attitude of doctors'] > 0.9).astype(int) |
                                                    (self.detailed_scores_df['attitude of nurses'] > 0.9).astype(int) |
                                                    (self.detailed_scores_df['doctor related'] == 1).astype(int))
        
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("doctors", case=False, na=False), 'Medical_Staff'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("nurse", case=False, na=False), 'Medical_Staff'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("dentist", case=False, na=False), 'Medical_Staff'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("staff", case=False, na=False), 'Medical_Staff'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("massage", case=False, na=False), 'Medical_Staff'] = 1

        self.detailed_scores_df['Receptionist_Front_Desk'] = ((self.detailed_scores_df['attitude of receptionist'] > 0.9).astype(int) |
                                                              (self.detailed_scores_df['Front desk related'] > 0.95).astype(int))

        self.detailed_scores_df['Facility Quality'] = ((self.detailed_scores_df['quality of building'] > 0.98).astype(int) |
                                                       (self.detailed_scores_df['place hygeine'] > 0.98).astype(int) |
                                                       (self.detailed_scores_df['cleanliness of hosptials'] > 0.98).astype(int))
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("ambience", case=False, na=False), 'Facility Quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("welcome environment", case=False, na=False), 'Facility Quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("clean", case=False, na=False), 'Facility Quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("facilities", case=False, na=False), 'Facility Quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("empathy", case=False, na=False), 'Facility Quality'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("hospital", case=False, na=False), 'Facility Quality'] = 1

        self.detailed_scores_df['Medication_Quality'] = (self.detailed_scores_df['medication quality'] > 0.95).astype(int)
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("treatment", case=False, na=False), 'Medication_Quality'] = 1

        self.detailed_scores_df['provider_wait_times'] = ((self.detailed_scores_df['time related'] == 1).astype(int) & (self.detailed_scores_df['medication_related'] == 1).astype(int))
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("delay", case=False, na=False), 'provider_wait_times'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("computerized process", case=False, na=False), 'provider_wait_times'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("code", case=False, na=False), 'provider_wait_times'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("approved", case=False, na=False), 'provider_wait_times'] = 1
        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("Timelineness", case=False, na=False), 'provider_wait_times'] = 1

        self.detailed_scores_df.loc[self.detailed_scores_df['CUSTOMER_RESPONSE'].str.contains("lack", case=False, na=False), 'Health_benefits_coverage'] = 1
        
        self.qa_data = self.detailed_scores_df[['CUSTOMER_RESPONSE', 'Health_benefits_coverage', 'provider_quality', 'RCC_quality',
                                                'provider_wait_times','customer_education','provider_network', 'tech_issues',
                                                'Medical_Staff', 'Receptionist_Front_Desk', 'Facility Quality', 'Medication_Quality', 
                                                'medication, treatment, or drug related']]
        
        self.nps_data_sample = self.nps_data_qa[['CUSTOMER_RESPONSE', 'type_service', 'provider_name']]
        self.nps_data_qa = pd.merge(self.nps_data_sample, self.qa_data, on='CUSTOMER_RESPONSE')
        
        def find_columns_with_one(row):
            cols = row.index[row == 1].tolist()
            return ', '.join(cols)
        
        self.nps_data_qa['main_topic'] = self.nps_data_qa[['Health_benefits_coverage', 'provider_quality', 'RCC_quality', 'provider_wait_times', 
                                                            'customer_education', 'provider_network', 'tech_issues']].apply(find_columns_with_one, axis=1)
        self.nps_data_qa['sub_topic'] = self.nps_data_qa[['Medical_Staff', 'Receptionist_Front_Desk', 'Facility Quality', 'Medication_Quality']].apply(find_columns_with_one, axis=1)
        
        self.final_df = self.nps_data_qa[['CUSTOMER_RESPONSE', 'main_topic', 'sub_topic', 'type_service']]
        self.final_df['sub_topic'] = self.final_df['sub_topic'].replace('', 'generic')
        print("Processing classifications completed.")
    
    def save_results(self, file_name):
        """
        Save the final results to a CSV file and upload to Snowflake.

        Parameters:
        file_name (str): The name of the file to save the results.
        """
        print(f"Saving results to {file_name}...")
        self.final_df.to_csv(file_name, index=False)
        upload_data(self.connection, self.final_df, 'NPS_QA_OP', if_exists='replace')
        print("Results saved and uploaded successfully.")

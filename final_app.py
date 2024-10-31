import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tempfile
import os
from groq import Groq
import nest_asyncio
import pytesseract
from pdf2image import convert_from_path
import cv2
import re
import plotly.express as px
import plotly.graph_objects as go

# Apply nest_asyncio to allow async operations in Streamlit
nest_asyncio.apply()

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to get binary image
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    
    return denoised

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using Tesseract OCR"""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        combined_text = ""
        for i, image in enumerate(images):
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Configure Tesseract parameters for better table recognition
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            # Extract text
            page_text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Clean up the extracted text
            page_text = re.sub(r'\s+', ' ', page_text)
            page_text = page_text.strip()
            
            if i > 0:
                combined_text += "\n\n--- Page Break ---\n\n"
            combined_text += page_text
        
        return combined_text
        
    except Exception as e:
        st.error(f"Error in Tesseract OCR: {str(e)}")
        return ""

def process_extracted_text(ocr_text: str, groq_api_key: str) -> str:
    """Process OCR output using Groq LLM"""
    client = Groq(api_key=groq_api_key)
    
    prompt = f"""
    You are an expert financial document analyzer. You have received text extracted from a bank statement using OCR. 
    Your task is to analyze the text and produce an accurate and complete transaction record.

    EXTRACTED TEXT:
    {ocr_text}

    TASK:
    Analyze the text and create a complete, accurate record of all transactions and account information.
    1. Use consistent date formats (MM/DD/YYYY)
    2. Verify transaction descriptions
    3. Ensure running balances are consistent
    4. Remove any duplicates
    5. Format all amounts with 2 decimal places

    REQUIRED OUTPUT FORMAT:

    Client Name: [Extracted name]
    Bank Name: [Extracted name]
    Account Number: [Extracted number]
    Statement Period: [Extracted period]

    Total Transactions: [Total number of transactions]

    Transaction Date | Credit/Debit | Description | Amount | Balance
    [MM/DD/YYYY] | [Credit/Debit] | [Description] | [Amount with 2 decimals] | [Balance with 2 decimals]

    CRITICAL RULES:
    1. List ALL transactions in chronological order
    2. Use exact pipe character '|' with single space on each side
    3. Format all dates as MM/DD/YYYY
    4. Format all amounts with 2 decimal places (no currency symbols)
    5. Include complete transaction descriptions
    6. Ensure running balances are mathematically consistent
    7. Mark any uncertain entries with '[UNVERIFIED]' in the description
    8. Remove any duplicate transactions
    9. Validate all numerical data for accuracy

    Format the output exactly as specified above with no additional commentary or markdown formatting.
    """

    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        model="llama3-8b-8192",
        max_tokens=8192,
        temperature=0.0,
        top_p=0.9
    )
    
    return chat_completion.choices[0].message.content

def parse_llm_output(output):
    """Parse the LLM output into summary and transactions"""
    lines = output.strip().split('\n')
    summary = {}
    transactions = []
    parsing_transactions = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line and not parsing_transactions:
            key, value = line.split(':', 1)
            summary[key.strip()] = value.strip()
        elif '|' in line and len(line.split('|')) == 5:
            parsing_transactions = True
            parts = [part.strip() for part in line.split('|')]
            try:
                transaction = {
                    'Transaction Date': parts[0],
                    'Credit/Debit': parts[1],
                    'Description': parts[2],
                    'Amount': float(parts[3].replace(',', '')),
                    'Balance': float(parts[4].replace(',', '')),
                    'Account Number': summary.get('Account Number', 'N/A')
                }
                transactions.append(transaction)
            except ValueError:
                # Skip header row or any row that can't be parsed
                continue

    return summary, transactions

def detect_fraudulent_transactions(df):
    """Detect potential fraudulent transactions using Isolation Forest"""
    # Prepare features for anomaly detection
    features = ['Amount']
    X = df[features].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X_scaled)
    
    # Add predictions to dataframe
    df['Is_Anomaly'] = clf.predict(X_scaled)
    df['Is_Anomaly'] = df['Is_Anomaly'].map({1: 'Normal', -1: 'Potential Fraud'})
    
    return df

def main():
    st.set_page_config(layout="wide", page_title="Bank Statement Analyzer")

    st.title("🏦 Bank Statement Analyzer with Fraud Detection")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_pdf = st.file_uploader("📄 Upload a bank statement PDF", type="pdf")
    
    with col2:
        st.info("This tool analyzes bank statements, extracts transactions, and detects potential fraud.")

    groq_api_key = st.text_input("Enter your Groq API Key", type="password")

    if uploaded_pdf is not None and groq_api_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_pdf.getvalue())
            pdf_path = tmp_file.name
        
        try:
            # Step 1: Extract text using OCR
            with st.spinner("Extracting text from PDF (this may take a minute)..."):
                ocr_text = extract_text_from_pdf(pdf_path)
            
            if not ocr_text:
                st.error("No content could be extracted from the PDF")
                return
            
            # Step 2: Process with Groq
            with st.spinner("Processing extracted text..."):
                processed_output = process_extracted_text(ocr_text, groq_api_key)
            
            # Parse output and display analyzed data
            with st.spinner("Parsing transactions..."):
                summary, transactions = parse_llm_output(processed_output)
            
            if transactions:
                df_transactions = pd.DataFrame(transactions)
                
                # Add Client Name and Bank Name columns
                df_transactions['Client Name'] = summary.get('Client Name', 'N/A')
                df_transactions['Bank Name'] = summary.get('Bank Name', 'N/A')
                
                # Reorder columns to match the desired format
                column_order = ['Client Name', 'Bank Name', 'Account Number', 'Transaction Date', 'Credit/Debit', 'Description', 'Amount', 'Balance']
                df_transactions = df_transactions.reindex(columns=column_order)
                
                st.success(f"✅ Successfully processed {len(df_transactions)} transactions!")
                
                st.subheader("📊 Account Summary")
                col1, col2 = st.columns(2)
                
                # Split the summary items into two halves
                summary_items = list(summary.items())
                mid_point = len(summary_items) // 2
                
                with col1:
                    for key, value in summary_items[:mid_point]:
                        st.metric(label=key, value=value)
                
                with col2:
                    for key, value in summary_items[mid_point:]:
                        st.metric(label=key, value=value)
                
                st.subheader("🧾 Transactions")
                st.dataframe(df_transactions.style.highlight_max(axis=0, subset=['Amount']), height=400)
                
                if not df_transactions.empty:
                    with st.spinner("Analyzing transactions for anomalies..."):
                        df_with_fraud = detect_fraudulent_transactions(df_transactions)
                    
                    st.subheader("🚨 Fraud Detection Results")
                    fraudulent_transactions = df_with_fraud[df_with_fraud['Is_Anomaly'] == 'Potential Fraud']
                    st.warning(f"Detected {len(fraudulent_transactions)} potentially fraudulent transactions.")
                    
                    if not fraudulent_transactions.empty:
                        st.dataframe(fraudulent_transactions.style.highlight_max(axis=0, subset=['Amount']), height=200)
                    else:
                        st.success("No potential fraudulent transactions detected.")
                    
                    # Download options
                    csv = df_with_fraud.to_csv(index=False)
                    st.download_button(
                        label="📥 Download transactions as CSV",
                        data=csv,
                        file_name="transactions_with_fraud_detection.csv",
                        mime="text/csv",
                    )
                
                st.subheader("📈 Transaction Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    # Bar chart of transaction amounts
                    fig = px.bar(df_transactions, x='Transaction Date', y='Amount', color='Credit/Debit',
                                 title='Transaction Amounts Over Time')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Pie chart of credit vs debit
                    credit_debit_sum = df_transactions.groupby('Credit/Debit')['Amount'].sum()
                    fig = px.pie(values=credit_debit_sum.values, names=credit_debit_sum.index,
                                 title='Credit vs Debit Distribution')
                    st.plotly_chart(fig, use_container_width=True)

                # Line chart of running balance
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_transactions['Transaction Date'], y=df_transactions['Balance'],
                                         mode='lines+markers', name='Running Balance'))
                fig.update_layout(title='Running Balance Over Time', xaxis_title='Date', yaxis_title='Balance')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No transactions found in the statement.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            os.unlink(pdf_path)
    elif uploaded_pdf and not groq_api_key:
        st.warning("Please enter the Groq API key first.")

    st.markdown("---")
    st.markdown("📚 Created for educational purposes. Not for actual financial advice.")

if __name__ == "__main__":
    main()

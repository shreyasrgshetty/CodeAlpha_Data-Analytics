# Install required libraries:
# pip install pandas faker

import pandas as pd
import random
from datetime import datetime, timedelta
import json

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    print("Installing faker library...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'faker'])
    from faker import Faker
    fake = Faker()

# ============================================
# DATASET 1: E-Commerce Products
# ============================================

def generate_ecommerce_dataset(num_records=100):
    """Generate e-commerce product dataset"""
    print("\n=== GENERATING E-COMMERCE DATASET ===")
    
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports', 'Toys']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
    ratings = [1, 2, 3, 4, 5]
    
    data = []
    for i in range(num_records):
        category = random.choice(categories)
        data.append({
            'product_id': f'PRD{i+1000}',
            'product_name': fake.catch_phrase(),
            'category': category,
            'brand': random.choice(brands),
            'price': round(random.uniform(10, 500), 2),
            'rating': random.choice(ratings),
            'reviews_count': random.randint(0, 1000),
            'in_stock': random.choice([True, False]),
            'discount_percent': random.choice([0, 5, 10, 15, 20, 25, 30]),
            'date_added': fake.date_between(start_date='-2y', end_date='today')
        })
    
    df = pd.DataFrame(data)
    df['final_price'] = df.apply(lambda x: round(x['price'] * (1 - x['discount_percent']/100), 2), axis=1)
    
    df.to_csv('ecommerce_products_dataset.csv', index=False)
    print(f"✓ Generated {num_records} product records")
    print(df.head())
    return df

# ============================================
# DATASET 2: Customer Data
# ============================================

def generate_customer_dataset(num_records=200):
    """Generate customer information dataset"""
    print("\n=== GENERATING CUSTOMER DATASET ===")
    
    data = []
    for i in range(num_records):
        data.append({
            'customer_id': f'CUST{i+10000}',
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'age': random.randint(18, 75),
            'gender': random.choice(['Male', 'Female', 'Other']),
            'city': fake.city(),
            'state': fake.state(),
            'country': fake.country(),
            'registration_date': fake.date_between(start_date='-3y', end_date='today'),
            'total_purchases': random.randint(0, 50),
            'total_spent': round(random.uniform(0, 5000), 2),
            'loyalty_status': random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv('customer_dataset.csv', index=False)
    df.to_json('customer_dataset.json', orient='records', indent=2)
    print(f"✓ Generated {num_records} customer records")
    print(df.head())
    return df

# ============================================
# DATASET 3: Sales Transactions
# ============================================

def generate_sales_dataset(num_records=500):
    """Generate sales transaction dataset"""
    print("\n=== GENERATING SALES DATASET ===")
    
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']
    statuses = ['Completed', 'Pending', 'Cancelled', 'Refunded']
    
    data = []
    for i in range(num_records):
        quantity = random.randint(1, 5)
        unit_price = round(random.uniform(10, 300), 2)
        
        data.append({
            'transaction_id': f'TXN{i+100000}',
            'customer_id': f'CUST{random.randint(10000, 10199)}',
            'product_id': f'PRD{random.randint(1000, 1099)}',
            'quantity': quantity,
            'unit_price': unit_price,
            'total_amount': round(quantity * unit_price, 2),
            'payment_method': random.choice(payment_methods),
            'status': random.choice(statuses),
            'transaction_date': fake.date_time_between(start_date='-1y', end_date='now'),
            'shipping_cost': round(random.uniform(0, 20), 2)
        })
    
    df = pd.DataFrame(data)
    df['grand_total'] = df['total_amount'] + df['shipping_cost']
    df.to_csv('sales_transactions_dataset.csv', index=False)
    print(f"✓ Generated {num_records} transaction records")
    print(df.head())
    return df

# ============================================
# DATASET 4: Employee Records
# ============================================

def generate_employee_dataset(num_records=150):
    """Generate employee HR dataset"""
    print("\n=== GENERATING EMPLOYEE DATASET ===")
    
    departments = ['IT', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    positions = ['Manager', 'Senior', 'Junior', 'Intern', 'Lead']
    
    data = []
    for i in range(num_records):
        hire_date = fake.date_between(start_date='-10y', end_date='today')
        
        data.append({
            'employee_id': f'EMP{i+5000}',
            'full_name': fake.name(),
            'email': fake.company_email(),
            'department': random.choice(departments),
            'position': random.choice(positions),
            'salary': random.randint(30000, 150000),
            'hire_date': hire_date,
            'years_of_service': (datetime.now().date() - hire_date).days // 365,
            'performance_score': round(random.uniform(1, 5), 2),
            'is_remote': random.choice([True, False]),
            'manager_id': f'EMP{random.randint(5000, 5020)}' if random.random() > 0.1 else None
        })
    
    df = pd.DataFrame(data)
    df.to_csv('employee_records_dataset.csv', index=False)
    df.to_excel('employee_records_dataset.xlsx', index=False)
    print(f"✓ Generated {num_records} employee records")
    print(df.head())
    return df

# ============================================
# DATASET 5: Social Media Posts
# ============================================

def generate_social_media_dataset(num_records=300):
    """Generate social media posts dataset"""
    print("\n=== GENERATING SOCIAL MEDIA DATASET ===")
    
    platforms = ['Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'TikTok']
    post_types = ['Text', 'Image', 'Video', 'Link', 'Poll']
    
    data = []
    for i in range(num_records):
        likes = random.randint(0, 10000)
        
        data.append({
            'post_id': f'POST{i+200000}',
            'user_id': f'USER{random.randint(1000, 5000)}',
            'username': fake.user_name(),
            'platform': random.choice(platforms),
            'post_type': random.choice(post_types),
            'content': fake.text(max_nb_chars=200),
            'hashtags': ', '.join([f'#{fake.word()}' for _ in range(random.randint(0, 5))]),
            'likes': likes,
            'comments': random.randint(0, likes//10),
            'shares': random.randint(0, likes//20),
            'views': likes * random.randint(5, 20),
            'posted_date': fake.date_time_between(start_date='-6m', end_date='now'),
            'is_verified': random.choice([True, False])
        })
    
    df = pd.DataFrame(data)
    df['engagement_rate'] = ((df['likes'] + df['comments'] + df['shares']) / df['views'] * 100).round(2)
    df.to_csv('social_media_posts_dataset.csv', index=False)
    print(f"✓ Generated {num_records} social media posts")
    print(df.head())
    return df

# ============================================
# DATASET 6: Website Analytics
# ============================================

def generate_website_analytics_dataset(num_records=1000):
    """Generate website traffic analytics dataset"""
    print("\n=== GENERATING WEBSITE ANALYTICS DATASET ===")
    
    pages = ['/home', '/products', '/about', '/contact', '/blog', '/checkout']
    devices = ['Desktop', 'Mobile', 'Tablet']
    browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
    sources = ['Direct', 'Google', 'Social Media', 'Email', 'Referral']
    
    data = []
    for i in range(num_records):
        data.append({
            'session_id': f'SES{i+300000}',
            'user_id': f'USER{random.randint(1000, 3000)}',
            'page_url': random.choice(pages),
            'visit_date': fake.date_time_between(start_date='-3m', end_date='now'),
            'device': random.choice(devices),
            'browser': random.choice(browsers),
            'traffic_source': random.choice(sources),
            'session_duration_sec': random.randint(10, 1800),
            'pages_viewed': random.randint(1, 15),
            'bounce': random.choice([True, False]),
            'conversion': random.choice([True, False]),
            'country': fake.country(),
            'city': fake.city()
        })
    
    df = pd.DataFrame(data)
    df['session_duration_min'] = (df['session_duration_sec'] / 60).round(2)
    df.to_csv('website_analytics_dataset.csv', index=False)
    print(f"✓ Generated {num_records} analytics records")
    print(df.head())
    return df

# ============================================
# DATASET 7: Weather Data
# ============================================

def generate_weather_dataset(num_records=365):
    """Generate weather data for a year"""
    print("\n=== GENERATING WEATHER DATASET ===")
    
    conditions = ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Snowy', 'Foggy']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    data = []
    start_date = datetime.now() - timedelta(days=num_records)
    
    for i in range(num_records):
        date = start_date + timedelta(days=i)
        city = random.choice(cities)
        
        data.append({
            'date': date.date(),
            'city': city,
            'temperature_f': round(random.uniform(20, 95), 1),
            'humidity': random.randint(20, 90),
            'wind_speed_mph': round(random.uniform(0, 30), 1),
            'precipitation_inch': round(random.uniform(0, 2), 2),
            'condition': random.choice(conditions),
            'visibility_miles': round(random.uniform(1, 10), 1),
            'pressure_inHg': round(random.uniform(29, 31), 2),
            'uv_index': random.randint(0, 11)
        })
    
    df = pd.DataFrame(data)
    df['temperature_c'] = ((df['temperature_f'] - 32) * 5/9).round(1)
    df.to_csv('weather_data_dataset.csv', index=False)
    print(f"✓ Generated {num_records} weather records")
    print(df.head())
    return df

# ============================================
# DATASET 8: Student Performance
# ============================================

def generate_student_dataset(num_records=250):
    """Generate student academic performance dataset"""
    print("\n=== GENERATING STUDENT DATASET ===")
    
    majors = ['Computer Science', 'Business', 'Engineering', 'Arts', 'Science']
    grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'D', 'F']
    
    data = []
    for i in range(num_records):
        data.append({
            'student_id': f'STU{i+20000}',
            'name': fake.name(),
            'age': random.randint(18, 25),
            'major': random.choice(majors),
            'year': random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior']),
            'gpa': round(random.uniform(2.0, 4.0), 2),
            'credits_completed': random.randint(0, 120),
            'attendance_rate': random.randint(60, 100),
            'grade': random.choice(grades),
            'scholarship': random.choice([True, False]),
            'extracurricular_activities': random.randint(0, 5),
            'study_hours_per_week': random.randint(5, 40)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('student_performance_dataset.csv', index=False)
    df.to_excel('student_performance_dataset.xlsx', index=False)
    print(f"✓ Generated {num_records} student records")
    print(df.head())
    return df

# ============================================
# MAIN EXECUTION WITH SUMMARY
# ============================================

def generate_dataset_summary(dataframes_dict):
    """Create a summary report of all generated datasets"""
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    summary_data = []
    for name, df in dataframes_dict.items():
        summary_data.append({
            'Dataset Name': name,
            'Total Records': len(df),
            'Total Columns': len(df.columns),
            'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'Missing Values': df.isnull().sum().sum()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('_DATASET_SUMMARY.csv', index=False)
    
    print("\n" + "="*70)
    print("DATASET GENERATION SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    """Generate all datasets automatically"""
    print("="*70)
    print("AUTOMATIC DATASET GENERATOR")
    print("Generating realistic datasets - No internet required!")
    print("="*70)
    
    try:
        # Generate all datasets
        dataframes = {
            'E-Commerce Products': generate_ecommerce_dataset(100),
            'Customers': generate_customer_dataset(200),
            'Sales Transactions': generate_sales_dataset(500),
            'Employees': generate_employee_dataset(150),
            'Social Media Posts': generate_social_media_dataset(300),
            'Website Analytics': generate_website_analytics_dataset(1000),
            'Weather Data': generate_weather_dataset(365),
            'Student Performance': generate_student_dataset(250)
        }
        
        # Generate summary
        summary = generate_dataset_summary(dataframes)
        
        print("\n" + "="*70)
        print("✓ ALL DATASETS GENERATED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated Files:")
        print("1. ecommerce_products_dataset.csv")
        print("2. customer_dataset.csv & customer_dataset.json")
        print("3. sales_transactions_dataset.csv")
        print("4. employee_records_dataset.csv & employee_records_dataset.xlsx")
        print("5. social_media_posts_dataset.csv")
        print("6. website_analytics_dataset.csv")
        print("7. weather_data_dataset.csv")
        print("8. student_performance_dataset.csv & student_performance_dataset.xlsx")
        print("9. _DATASET_SUMMARY.csv (Overview of all datasets)")
        print("\nTotal Records Generated:", sum(len(df) for df in dataframes.values()))
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if _name_ == "_main_":
    main()
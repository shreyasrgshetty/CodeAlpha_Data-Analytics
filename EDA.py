# Install required libraries:
# pip install pandas numpy matplotlib seaborn faker scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import random

warnings.filterwarnings('ignore')

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'faker'])
    from faker import Faker
    fake = Faker()

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# STEP 1: GENERATE DATASET
# ============================================

def generate_sales_dataset(num_records=1000):
    """Generate comprehensive sales dataset for analysis"""
    print("="*70)
    print("STEP 1: GENERATING DATASET")
    print("="*70)
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys']
    regions = ['North', 'South', 'East', 'West', 'Central']
    customer_types = ['Regular', 'Premium', 'VIP']
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash']
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_records):
        category = random.choice(categories)
        quantity = random.randint(1, 10)
        base_price = random.uniform(10, 500)
        discount = random.choice([0, 5, 10, 15, 20, 25])
        
        # Introduce some missing values (realistic scenario)
        customer_age = random.randint(18, 75) if random.random() > 0.05 else None
        rating = random.randint(1, 5) if random.random() > 0.1 else None
        
        data.append({
            'order_id': f'ORD{i+10000}',
            'date': start_date + timedelta(days=random.randint(0, 365)),
            'category': category,
            'product_name': fake.catch_phrase(),
            'quantity': quantity,
            'unit_price': round(base_price, 2),
            'discount_percent': discount,
            'total_price': round(quantity * base_price * (1 - discount/100), 2),
            'customer_type': random.choice(customer_types),
            'customer_age': customer_age,
            'region': random.choice(regions),
            'payment_method': random.choice(payment_methods),
            'rating': rating,
            'shipping_days': random.randint(1, 14),
            'returned': random.choice([True, False]) if random.random() < 0.1 else False
        })
    
    df = pd.DataFrame(data)
    df.to_csv('sales_data_for_eda.csv', index=False)
    
    print(f"✓ Generated {num_records} records")
    print(f"✓ Saved to: sales_data_for_eda.csv")
    print(f"\nDataset Preview:")
    print(df.head())
    
    return df

# ============================================
# STEP 2: ASK MEANINGFUL QUESTIONS
# ============================================

def ask_meaningful_questions(df):
    """Define key questions to guide the analysis"""
    print("\n" + "="*70)
    print("STEP 2: MEANINGFUL QUESTIONS ABOUT THE DATASET")
    print("="*70)
    
    questions = [
        "1. What is the overall sales trend over time?",
        "2. Which product categories generate the most revenue?",
        "3. How does customer type affect purchase behavior?",
        "4. What is the relationship between discount and sales volume?",
        "5. Which regions have the highest sales performance?",
        "6. What is the average customer rating by category?",
        "7. Is there a correlation between customer age and spending?",
        "8. What percentage of orders are returned?",
        "9. Which payment methods are most popular?",
        "10. How does shipping time affect customer ratings?"
    ]
    
    for q in questions:
        print(f"❓ {q}")
    
    return questions

# ============================================
# STEP 3: EXPLORE DATA STRUCTURE
# ============================================

def explore_data_structure(df):
    """Comprehensive data structure exploration"""
    print("\n" + "="*70)
    print("STEP 3: EXPLORING DATA STRUCTURE")
    print("="*70)
    
    print("\n📊 Dataset Shape:")
    print(f"   Rows: {df.shape[0]}")
    print(f"   Columns: {df.shape[1]}")
    
    print("\n📋 Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n📈 Numerical Columns Summary:")
    print(df.describe())
    
    print("\n📝 Categorical Columns Summary:")
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        if col != 'order_id' and col != 'product_name':
            print(f"\n{col}:")
            print(df[col].value_counts())
    
    print("\n💾 Memory Usage:")
    print(f"   Total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save structure report
    with open('data_structure_report.txt', 'w') as f:
        f.write("DATA STRUCTURE REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("Summary Statistics:\n")
        f.write(str(df.describe()) + "\n")
    
    print("\n✓ Structure report saved to: data_structure_report.txt")

# ============================================
# STEP 4: IDENTIFY TRENDS, PATTERNS & ANOMALIES
# ============================================

def identify_trends_patterns(df):
    """Identify key trends, patterns, and anomalies"""
    print("\n" + "="*70)
    print("STEP 4: IDENTIFYING TRENDS, PATTERNS & ANOMALIES")
    print("="*70)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df['day_of_week'] = df['date'].dt.day_name()
    
    print("\n📈 TREND 1: Sales Over Time")
    monthly_sales = df.groupby('month')['total_price'].sum()
    print(monthly_sales)
    
    print("\n📊 PATTERN 1: Sales by Category")
    category_sales = df.groupby('category')['total_price'].sum().sort_values(ascending=False)
    print(category_sales)
    
    print("\n🔍 PATTERN 2: Sales by Day of Week")
    dow_sales = df.groupby('day_of_week')['total_price'].mean().sort_values(ascending=False)
    print(dow_sales)
    
    print("\n⚠  ANOMALY DETECTION:")
    
    # Detect outliers using IQR method
    Q1 = df['total_price'].quantile(0.25)
    Q3 = df['total_price'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['total_price'] < Q1 - 1.5 * IQR) | (df['total_price'] > Q3 + 1.5 * IQR)]
    
    print(f"   • Found {len(outliers)} outlier transactions (extreme prices)")
    print(f"   • Outlier range: ${outliers['total_price'].min():.2f} - ${outliers['total_price'].max():.2f}")
    
    # High return rate products
    return_rate = df.groupby('category')['returned'].mean() * 100
    high_return = return_rate[return_rate > 15]
    if len(high_return) > 0:
        print(f"   • Categories with high return rates (>15%):")
        for cat, rate in high_return.items():
            print(f"     - {cat}: {rate:.1f}%")
    
    # Save trends report
    trends_report = pd.DataFrame({
        'Monthly Sales': monthly_sales,
        'Category Sales': category_sales
    })
    trends_report.to_csv('trends_analysis.csv')
    print("\n✓ Trends saved to: trends_analysis.csv")
    
    return df

# ============================================
# STEP 5: TEST HYPOTHESES & VALIDATE ASSUMPTIONS
# ============================================

def test_hypotheses(df):
    """Test statistical hypotheses and validate assumptions"""
    print("\n" + "="*70)
    print("STEP 5: TESTING HYPOTHESES & VALIDATING ASSUMPTIONS")
    print("="*70)
    
    from scipy import stats
    
    # Hypothesis 1: Premium customers spend more than regular customers
    print("\n🔬 HYPOTHESIS 1: Premium customers spend more than Regular customers")
    premium_spending = df[df['customer_type'] == 'Premium']['total_price']
    regular_spending = df[df['customer_type'] == 'Regular']['total_price']
    
    t_stat, p_value = stats.ttest_ind(premium_spending, regular_spending)
    print(f"   Premium avg: ${premium_spending.mean():.2f}")
    print(f"   Regular avg: ${regular_spending.mean():.2f}")
    print(f"   T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("   ✓ RESULT: Statistically significant difference (p < 0.05)")
    else:
        print("   ✗ RESULT: No significant difference (p >= 0.05)")
    
    # Hypothesis 2: Discounts increase sales volume
    print("\n🔬 HYPOTHESIS 2: Discounts affect sales volume")
    with_discount = df[df['discount_percent'] > 0]['quantity'].mean()
    without_discount = df[df['discount_percent'] == 0]['quantity'].mean()
    
    print(f"   Avg quantity with discount: {with_discount:.2f}")
    print(f"   Avg quantity without discount: {without_discount:.2f}")
    print(f"   Difference: {(with_discount - without_discount):.2f} units")
    
    # Correlation Analysis
    print("\n📊 CORRELATION ANALYSIS:")
    numeric_cols = ['quantity', 'unit_price', 'discount_percent', 'total_price', 
                    'customer_age', 'rating', 'shipping_days']
    correlation_matrix = df[numeric_cols].corr()
    
    print("\nStrong Correlations (|r| > 0.5):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"   • {correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_val:.3f}")
    
    # Assumption: Data is normally distributed
    print("\n📈 NORMALITY TEST (Shapiro-Wilk):")
    stat, p_value = stats.shapiro(df['total_price'].sample(min(5000, len(df))))
    print(f"   Total Price: p-value = {p_value:.4f}")
    if p_value > 0.05:
        print("   ✓ Data appears normally distributed")
    else:
        print("   ✗ Data is not normally distributed (common in real data)")
    
    correlation_matrix.to_csv('correlation_analysis.csv')
    print("\n✓ Correlation matrix saved to: correlation_analysis.csv")

# ============================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================

def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("\n" + "="*70)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Sales Trend Over Time
    ax1 = plt.subplot(3, 3, 1)
    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['total_price'].sum()
    monthly_sales.plot(kind='line', marker='o', ax=ax1)
    ax1.set_title('Sales Trend Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    
    # 2. Sales by Category
    ax2 = plt.subplot(3, 3, 2)
    category_sales = df.groupby('category')['total_price'].sum().sort_values()
    category_sales.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Sales by Category', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Sales ($)')
    
    # 3. Customer Type Distribution
    ax3 = plt.subplot(3, 3, 3)
    df['customer_type'].value_counts().plot(kind='pie', ax=ax3, autopct='%1.1f%%')
    ax3.set_title('Customer Type Distribution', fontsize=12, fontweight='bold')
    ax3.set_ylabel('')
    
    # 4. Price Distribution
    ax4 = plt.subplot(3, 3, 4)
    df['total_price'].hist(bins=50, ax=ax4, edgecolor='black', alpha=0.7)
    ax4.set_title('Price Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Total Price ($)')
    ax4.set_ylabel('Frequency')
    
    # 5. Correlation Heatmap
    ax5 = plt.subplot(3, 3, 5)
    numeric_cols = ['quantity', 'unit_price', 'discount_percent', 'total_price', 'rating', 'shipping_days']
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax5, center=0)
    ax5.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')
    
    # 6. Sales by Region
    ax6 = plt.subplot(3, 3, 6)
    region_sales = df.groupby('region')['total_price'].sum().sort_values()
    region_sales.plot(kind='bar', ax=ax6, color='coral')
    ax6.set_title('Sales by Region', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Region')
    ax6.set_ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    
    # 7. Rating Distribution
    ax7 = plt.subplot(3, 3, 7)
    df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax7, color='green')
    ax7.set_title('Customer Ratings Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Rating')
    ax7.set_ylabel('Count')
    
    # 8. Discount vs Quantity
    ax8 = plt.subplot(3, 3, 8)
    discount_groups = df.groupby('discount_percent')['quantity'].mean()
    discount_groups.plot(kind='line', marker='o', ax=ax8, color='purple')
    ax8.set_title('Discount % vs Avg Quantity', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Discount %')
    ax8.set_ylabel('Avg Quantity')
    
    # 9. Return Rate by Category
    ax9 = plt.subplot(3, 3, 9)
    return_rate = df.groupby('category')['returned'].mean() * 100
    return_rate.plot(kind='bar', ax=ax9, color='red', alpha=0.7)
    ax9.set_title('Return Rate by Category', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Category')
    ax9.set_ylabel('Return Rate (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to: eda_visualizations.png")
    plt.show()

# ============================================
# STEP 7: DETECT DATA ISSUES
# ============================================

def detect_data_issues(df):
    """Detect and report data quality issues"""
    print("\n" + "="*70)
    print("STEP 7: DETECTING DATA ISSUES & PROBLEMS")
    print("="*70)
    
    issues = []
    
    # Missing Values
    print("\n🔍 MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    for col in missing[missing > 0].index:
        pct = missing_pct[col]
        print(f"   • {col}: {missing[col]} ({pct:.2f}%)")
        issues.append(f"Missing values in {col}: {missing[col]} records ({pct:.2f}%)")
    
    if missing.sum() == 0:
        print("   ✓ No missing values detected")
    
    # Duplicates
    print("\n🔍 DUPLICATE RECORDS:")
    duplicates = df.duplicated().sum()
    print(f"   • Found {duplicates} duplicate rows")
    if duplicates > 0:
        issues.append(f"Duplicate records: {duplicates}")
    else:
        print("   ✓ No duplicates found")
    
    # Outliers
    print("\n🔍 OUTLIERS:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        if len(outliers) > 0:
            print(f"   • {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
            issues.append(f"Outliers in {col}: {len(outliers)} values")
    
    # Inconsistent Data
    print("\n🔍 DATA CONSISTENCY:")
    
    # Check for negative values where they shouldn't exist
    if (df['total_price'] < 0).any():
        neg_count = (df['total_price'] < 0).sum()
        print(f"   • Negative prices found: {neg_count}")
        issues.append(f"Negative prices: {neg_count} records")
    
    if (df['quantity'] <= 0).any():
        zero_qty = (df['quantity'] <= 0).sum()
        print(f"   • Zero/negative quantities: {zero_qty}")
        issues.append(f"Invalid quantities: {zero_qty} records")
    
    if len(issues) == 0:
        print("   ✓ No consistency issues detected")
    
    # Data Quality Score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    quality_score = ((total_cells - missing_cells) / total_cells) * 100
    
    print(f"\n📊 DATA QUALITY SCORE: {quality_score:.2f}%")
    
    # Save issues report
    with open('data_issues_report.txt', 'w') as f:
        f.write("DATA QUALITY ISSUES REPORT\n")
        f.write("="*50 + "\n\n")
        if len(issues) > 0:
            for issue in issues:
                f.write(f"• {issue}\n")
        else:
            f.write("✓ No major data quality issues detected\n")
        f.write(f"\nData Quality Score: {quality_score:.2f}%\n")
    
    print("✓ Issues report saved to: data_issues_report.txt")

# ============================================
# MAIN EDA PIPELINE
# ============================================

def main():
    """Execute complete EDA pipeline"""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA) - COMPLETE PIPELINE")
    print("="*70)
    print("Automatically generates data and performs comprehensive analysis")
    print("="*70 + "\n")
    
    try:
        # Step 1: Generate Dataset
        df = generate_sales_dataset(1000)
        
        # Step 2: Ask Questions
        questions = ask_meaningful_questions(df)
        
        # Step 3: Explore Structure
        explore_data_structure(df)
        
        # Step 4: Identify Trends
        df = identify_trends_patterns(df)
        
        # Step 5: Test Hypotheses
        test_hypotheses(df)
        
        # Step 6: Create Visualizations
        create_visualizations(df)
        
        # Step 7: Detect Issues
        detect_data_issues(df)
        
        print("\n" + "="*70)
        print("✓ COMPLETE EDA FINISHED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Generated Files:")
        print("   1. sales_data_for_eda.csv - Original dataset")
        print("   2. data_structure_report.txt - Structure analysis")
        print("   3. trends_analysis.csv - Trends and patterns")
        print("   4. correlation_analysis.csv - Correlation matrix")
        print("   5. eda_visualizations.png - All visualizations")
        print("   6. data_issues_report.txt - Quality issues")
        print("\n🎯 Analysis complete! Review the files and visualizations.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if name == "main":
    main()
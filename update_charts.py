with open('app.py', 'r') as file:
    content = file.read()

# Replace "safe_plotly_chart(fig)" with unique keys based on surrounding content

# 1. History trend chart (around line 364)
content = content.replace(
    'yaxis_range=[-1, 1]\n            )\n            \n            safe_plotly_chart(fig)',
    'yaxis_range=[-1, 1]\n            )\n            \n            safe_plotly_chart(fig, key="history_trend")'
)

# 2. Sentiment distribution pie chart (around line 484)
content = content.replace(
    'title="Sentiment Distribution"\n            )\n            \n            safe_plotly_chart(fig)',
    'title="Sentiment Distribution"\n            )\n            \n            safe_plotly_chart(fig, key="sentiment_distribution_pie")'
)

# 3. Sentiment histogram chart (around line 502)
content = content.replace(
    'xaxis_range=[-1, 1]\n            )\n            \n            safe_plotly_chart(fig)',
    'xaxis_range=[-1, 1]\n            )\n            \n            safe_plotly_chart(fig, key="sentiment_histogram")'
)

# 4. Emotion bar chart (analytics tab)
content = content.replace(
    'yaxis_range=[0, 1]\n            )\n            \n            safe_plotly_chart(fig)',
    'yaxis_range=[0, 1]\n            )\n            \n            safe_plotly_chart(fig, key="emotion_intensity_bar")'
)

# 5. Aspect frequency chart
content = content.replace(
    'margin=dict(l=50, r=50, t=50, b=50)\n            )\n            \n            safe_plotly_chart(fig)',
    'margin=dict(l=50, r=50, t=50, b=50)\n            )\n            \n            safe_plotly_chart(fig, key="aspect_frequency_chart")'
)

# 6. Aspect sentiment chart
content = content.replace(
    'yaxis_title="Count"\n            )\n            \n            safe_plotly_chart(fig)',
    'yaxis_title="Count"\n            )\n            \n            safe_plotly_chart(fig, key="aspect_sentiment_chart")'
)

# 7. Subjectivity scatter plot
content = content.replace(
    'yaxis_range=[0, 1]\n        )\n        \n        safe_plotly_chart(fig)',
    'yaxis_range=[0, 1]\n        )\n        \n        safe_plotly_chart(fig, key="subjectivity_scatter")'
)

with open('app.py', 'w') as file:
    file.write(content)

print("Updated all chart keys successfully!")

from flask import Flask, render_template, request, jsonify
from search_aggregator import SearchAggregator

app = Flask(__name__)
search_aggregator = SearchAggregator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    num_results = request.json.get('num_results', 5)
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    results = search_aggregator.aggregate_search(query, num_results)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True) 
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Provider } from 'react-redux';
import { Toaster } from 'react-hot-toast';

// Components
import Navbar from './components/Layout/Navbar';
import Footer from './components/Layout/Footer';

// Pages
import Home from './pages/Home';
import Login from './pages/Auth/Login';
import Register from './pages/Auth/Register';
import Dashboard from './pages/Dashboard';
import Practice from './pages/Practice';
import MockTests from './pages/MockTests';
import StudyPlans from './pages/StudyPlans';
import CurrentAffairs from './pages/CurrentAffairs';
import Pricing from './pages/Pricing';
import Marketplace from './pages/Marketplace';
import Profile from './pages/Profile';

function App() {
  return (
    <Provider store={store}>
      <Router>
        <div className="App">
          <Navbar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/practice" element={<Practice />} />
              <Route path="/mock-tests" element={<MockTests />} />
              <Route path="/study-plans" element={<StudyPlans />} />
              <Route path="/current-affairs" element={<CurrentAffairs />} />
              <Route path="/pricing" element={<Pricing />} />
              <Route path="/marketplace" element={<Marketplace />} />
              <Route path="/profile" element={<Profile />} />
            </Routes>
          </main>
          <Footer />
          <Toaster position="top-right" />
        </div>
      </Router>
    </Provider>
  );
}

export default App;
import { createContext, useContext, useState } from 'react';
import { ua } from '../locales/ua';
import { pl } from '../locales/pl';

const locales = { ua, pl };

const LanguageContext = createContext(null);

export function LanguageProvider({ children }) {
  const [language, setLanguage] = useState('ua');

  const t = locales[language];

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
